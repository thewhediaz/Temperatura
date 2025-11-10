import geopandas as gpd
import matplotlib.pyplot as plt
import requests
from datetime import timedelta, timezone,datetime
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.path import Path
from shapely.geometry import Polygon, MultiPolygon
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import matplotlib.colors as mcolors
import os
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pytz


# Configuración de la página
st.set_page_config(
    page_title="Protección Civil - Acapulco",
    layout="wide"
)

st.markdown(
    """
    <style>
    div[data-testid="stAppViewContainer"] .block-container {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
        padding-bottom: 0rem !important;
        margin-bottom: 0rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }

    header[data-testid="stHeader"] {
        display: none !important;
    }

    footer {
        visibility: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Crear cabecera con logos
col1, col2 = st.columns([1, 1])
with col1:
    st.image("https://acapulco.gob.mx/proteccioncivil/wp-content/uploads/2025/07/CGPCYB_24.png", width=300)
with col2:
    st.markdown(
        """
        <div style="text-align: right;">
            <img src="https://i.ibb.co/zTM1fBTg/SIATM.png" width="90">
        </div>
        """,
        unsafe_allow_html=True
    )
###############################################################################



# ---------------------------------------------------------------------------
# AUTOREFRESH: exactamente en cada cuarto de hora (00, 15, 30, 45)
ahora = datetime.now()
multiplo15 = (ahora.minute // 15) * 15
objetivo = ahora.replace(minute=multiplo15, second=10, microsecond=0)
if ahora >= objetivo:
    objetivo += timedelta(minutes=15)
delta = objetivo - ahora
milisegundos_hasta_refresh = int(delta.total_seconds() * 1000)
count = st_autorefresh(interval=milisegundos_hasta_refresh, key="datarefresh")






kk = 6
def rangos():
    """
    Obtiene las temperaturas mínima y máxima globales de una hoja de Google Sheets pública.
    Si han pasado menos de 24h desde la última ejecución, lee los valores previos del archivo cache.
    """
    cache_file = "rangos_cache.txt"
    cooldown = timedelta(hours=24)

    # --- 1. Verificar si existe cache y si aún está dentro del cooldown ---
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            lines = f.read().strip().split(",")
            if len(lines) == 3:
                last_time = datetime.fromisoformat(lines[0])
                tmin_prev = float(lines[1])
                tmax_prev = float(lines[2])

                # Si no han pasado 24 horas, usar los valores del cache
                if datetime.now() - last_time < cooldown:
                    return tmin_prev, tmax_prev

    # --- 2. Si no hay cache válido, ejecutar la descarga ---

    sheet_id = "1aL5PkK8J-1wI9RZk0nOFT9Cd35jZXPB96FVieh8QeOw"
    gids = [645419203, 394552055, 739103023, 912283735, 1707675775]

    def get_min_max(gid):
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
        df = pd.read_csv(url)
        col = [c for c in df.columns if "Temperatura" in c][0]
        df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[col].min(), df[col].max()

    with ThreadPoolExecutor(max_workers=len(gids)) as executor:
        results = list(executor.map(get_min_max, gids))

    min_temp = min(r[0] for r in results)
    max_temp = max(r[1] for r in results)

    # --- 3. Guardar resultados y hora actual en cache ---
    with open(cache_file, "w") as f:
        f.write(f"{datetime.now().isoformat()},{min_temp},{max_temp}")

    return min_temp, max_temp

minimo,maximo = rangos()

# CLAVES API OCULTAR
API_KEY = st.secrets["API_KEY"]

API_SECRET = st.secrets["API_SECRET"]
# CLAVES APi OCULTAR

station_ids = ["220125", "222236", "214340", "214736", "214739","222011"]

def obtener_datos_estaciones(ids):
    h = {'x-api-secret': API_SECRET}
    p = {'api-key': API_KEY}
    nombres, lat, lon, temp_c = [], [], [], []
    for i in ids:
        m = requests.get(f"https://api.weatherlink.com/v2/stations/{i}", params=p, headers=h)
        if m.status_code != 200: continue
        s = m.json().get('stations', [{}])[0]
        n, la, lo = s.get('station_name'), s.get('latitude'), s.get('longitude')
        d = requests.get(f"https://api.weatherlink.com/v2/current/{i}", params=p, headers=h)
        if d.status_code != 200: continue
        t = None
        for sen in d.json().get('sensors', []):
            for x in sen.get('data', []):
                t = x.get('temp') or x.get('temp_out')
            #   t = x.get('temp') or x.get('temp_out')
                if t is not None: break
            if t is not None: break
        if t is not None and t > 45: t = (t - 32) * 5/9
        nombres.append(n); lat.append(la); lon.append(lo); temp_c.append(t)
    return nombres, lat, lon, temp_c

nombres, lats, lons, temps_c = obtener_datos_estaciones(station_ids)

# Leer shapefile y reproyectar
aca = gpd.read_file("12mun.shp")
aca = aca.to_crs(epsg=4326)

# Filtrar Acapulco
acapulco = aca[aca["CVE_MUN"] == "001"]

# limites de acapulco
bounds = acapulco.total_bounds  
minx, miny, maxx, maxy = bounds 

localidades = gpd.read_file("localidades_acapulco.shp")

# ------------------------------

# --- Crear malla (grid) para interpolar ---
n = 400  # resolución de la cuadrícula
xi = np.linspace(minx, maxx, n)
yi = np.linspace(miny, maxy, n)
xi, yi = np.meshgrid(xi, yi)
coords_grid = np.vstack((xi.ravel(), yi.ravel())).T

# --- Interpolación IDW (Inverse Distance Weighting) ---
tree = cKDTree(np.vstack((lons, lats)).T)
dist, idx = tree.query(coords_grid, k=kk)
weights = 1 / (dist + 1e-10)**2
z_idw = np.sum(weights * np.take(temps_c, idx), axis=1) / np.sum(weights, axis=1)
zi = z_idw.reshape(xi.shape)

# --- Crear máscara del polígono de Acapulco para el fondo ---
mask_total = np.zeros(coords_grid.shape[0], dtype=bool)
geom = acapulco.geometry.iloc[0]
if isinstance(geom, Polygon):
    paths = [Path(np.array(geom.exterior.coords))]
elif isinstance(geom, MultiPolygon):
    paths = [Path(np.array(p.exterior.coords)) for p in geom.geoms]

for p in paths:
    mask_total |= p.contains_points(coords_grid)

mask = mask_total.reshape(xi.shape)

# --- Graficar mapa ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()

ax.set_facecolor("#AAD3E5")
aca.plot(ax=ax, color="burlywood", edgecolor='black', linewidth=0.5)
acapulco.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, zorder=3)
localidades.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, zorder=3)

espacio_extra = (maxx - minx) * 0.2
maxx_expandido = maxx + espacio_extra
ax.set_xlim(minx - 0.05, maxx_expandido)
ax.set_ylim(miny - 0.05, maxy + 0.05)

# --- Interpolación como fondo de color (solo dentro de Acapulco) ---
cmap = mcolors.LinearSegmentedColormap.from_list(
    "temp_cmap", ["blue","green","yellow", "red"]
)
zi_masked = np.ma.array(zi, mask=~mask)
im =ax.imshow(
    zi_masked,
    extent=(minx, maxx, miny, maxy),
    origin='lower',
    cmap=cmap,
    alpha=1,
    zorder=1,
    vmin=minimo,
    vmax=maximo

)

cbar = plt.colorbar(
    im, ax=ax,
    orientation='horizontal',
    fraction=0.02,   # ← más delgada (antes 0.05)
    pad=0.04,        # separación con el mapa (ligeramente ajustada)
    shrink=1,      # ← largo completo, sin recorte
    aspect =49.5,
    extend='both',   # agrega triángulos en los extremos
    extendfrac=0.06  # tamaño de los triángulos
)

# Asegurarnos de incluir mínimo y máximo y algunos intermedios
num_ticks = 6  # o los que quieras
ticks = np.linspace(minimo, maximo, num=num_ticks)
ticks_rounded = [int(round(t)) for t in ticks]  # redondear a enteros
cbar.set_ticks(ticks)
cbar.ax.set_xticklabels(ticks_rounded)  # mostrar los ticks redondeados



# agregamos las etiquetas para las localidades
plt.text(-99.718144, 17.082704, "Xaltianguis", fontsize=6, ha='center', va='center')
plt.text(-99.765889, 16.982827, "Kilómetro 30", fontsize=6, ha='center', va='center')
plt.text(-99.770345, 16.818687, "Tres palos", fontsize=6, ha='center', va='center')
plt.text(-99.734460, 16.801511, "San Pedro \n las playas", fontsize=6, ha='center', va='center')
plt.text(-99.664944, 16.808910, "Amatillo", fontsize=6, ha='center', va='center')
plt.text(-99.900813, 16.798789, "Acapulco", fontsize=6, ha='center', va='center')
plt.text(-99.762896, 16.855850, "El salto", fontsize=6, ha='center', va='center')
plt.text(-99.575054, 16.885015, "Huamuchitos", fontsize=6, ha='center', va='center')
plt.text(-99.604996, 16.705070, "Lomas de Chapultepec", fontsize=6, ha='center', va='center')
plt.text(-99.650092, 17.006250, "Dos Arroyos", fontsize=6, ha='center', va='center')
plt.text(-99.737185, 16.965498, "Ejido Nuevo", fontsize=6, ha='center', va='center')


# --- Líneas de igual temperatura sobre toda la cuadrícula ---
niveles = np.linspace(np.min(temps_c), np.max(temps_c), 10)
cont_lines = ax.contour(
    xi, yi, zi,  # <- sin máscara, para que salgan fuera de Acapulco
    levels=niveles, colors='black', linewidths=1, zorder=4
)
ax.clabel(cont_lines, inline=True, fontsize=8, fmt="%.1f°C")

x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

# --- Identificar la temperatura máxima ---
temp_max = max(temps_c)

for lon, lat, temp in zip(lons, lats, temps_c):
    # Solo dibujar si el punto está dentro del área visible
    if not (x_min <= lon <= x_max and y_min <= lat <= y_max):
        continue

    # Si es la estación con la temperatura más alta → fondo rojo
    if temp == temp_max:
        facecolor = 'red'
        alpha = 0.5
        fontcolor = 'white'
        fontsize = 9
        fontweight = 'bold'
    else:
        facecolor = 'black'
        alpha = 0.5
        fontcolor = 'white'
        fontsize = 8
        fontweight = 'bold'

    ax.text(
        lon, lat,
        f"{temp:.1f}°C",
        color=fontcolor,
        fontsize=fontsize,
        fontweight=fontweight,
        ha='center',
        va='center',
        zorder=5,
        bbox=dict(
            facecolor=facecolor,
            alpha=alpha,
            edgecolor='none',
            boxstyle='round,pad=0.2'
        )
    )


ax.set_title(
        "   Distribución Espacial de la Temperatura Instantánea del Aire\n", 
        fontsize=17,
        loc='left'
        )

# validez de los datos 
dias_semana = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
         "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
#  Obtener fecha y hora actual
ahora = datetime.now()
#  Redondear hacia abajo al cuarto de hora
minuto_redondeado = (ahora.minute // 15) * 15
fecha_redondeada = ahora.replace(minute=minuto_redondeado, second=0, microsecond=0)
# separamos
dia_semana = dias_semana[fecha_redondeada.weekday()]
dia = fecha_redondeada.day
mes = meses[fecha_redondeada.month - 1]

# fecha en am o pm
hora_12 = fecha_redondeada.strftime("%I:%M")
am_pm = "a.m." if fecha_redondeada.strftime("%p") == "AM" else "p.m."


ax.text(
        0, 1.01,  # posición (X=1 derecha, Y=ligeramente arriba)
        "Válido el " + dia_semana + " " + str(dia) + " de " + mes + " a las " + hora_12 + " " + am_pm,
        fontsize=11,
        ha='left',
        transform=ax.transAxes,
        )

ax.text(
        1, 1.01,  # posición (X=1 derecha, Y=ligeramente arriba)
        "Fuente de datos: SIATM-ACA; SIAT-GRO.", 
        fontsize=11,
        ha='right',
        transform=ax.transAxes,
        )

ax.text(
        1, 0.97,  # posición (X=1 derecha, Y=ligeramente arriba)
        "Superficie generada mediante interpolación IDW", 
        fontsize=11,
        ha='right',
        transform=ax.transAxes,
        )

logo_img = mpimg.imread("LOGO.png")
ab = AnnotationBbox(
    OffsetImage(logo_img, zoom=0.68),
    (-99.954, 16.68),         
    frameon=False,
    xycoords='data'        
)
ax.add_artist(ab)

logo_img2 = mpimg.imread("ROSA.png")
ab2 = AnnotationBbox(
    OffsetImage(logo_img2, zoom=0.6),
    (-99.441, 16.686),         
    frameon=False,
    xycoords='data'        
)
ax.add_artist(ab2)


###############################################################################
# Centrar gráfica en la página
st.pyplot(fig, use_container_width=True)

