import streamlit as st

try:
    
    import os
    import requests
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from datetime import datetime, timedelta
    from scipy.spatial import cKDTree
    from shapely.geometry import Polygon, MultiPolygon
    from matplotlib.path import Path
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import matplotlib.image as mpimg
    from streamlit_autorefresh import st_autorefresh
    
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
    objetivo = ahora.replace(minute=(multiplo15 + 1) % 60, second=10, microsecond=0)
    if ahora >= objetivo:
        objetivo += timedelta(minutes=15)
    delta = objetivo - ahora
    milisegundos_hasta_refresh = int(delta.total_seconds() * 1000)
    count = st_autorefresh(interval=milisegundos_hasta_refresh, key="datarefresh")
    
    
    
    
    
    
    kk = 6  # número de vecinos IDW
    
    # === RANGO DE HUMEDAD (FIJO 0–100%) ===
    minimo, maximo = 0, 100
    
    # === API KEYS ===
    API_KEY = st.secrets["API_KEY"]
    
    API_SECRET = st.secrets["API_SECRET"]
    
    station_ids = ["220125", "222236", "214340", "214736", "214739", "222011"]
    
    def obtener_datos_estaciones(ids):
        h = {'x-api-secret': API_SECRET}
        p = {'api-key': API_KEY}
        nombres, lat, lon, hum_c = [], [], [], []
        for i in ids:
            m = requests.get(f"https://api.weatherlink.com/v2/stations/{i}", params=p, headers=h)
            if m.status_code != 200:
                continue
            s = m.json().get('stations', [{}])[0]
            n, la, lo = s.get('station_name'), s.get('latitude'), s.get('longitude')
            d = requests.get(f"https://api.weatherlink.com/v2/current/{i}", params=p, headers=h)
            if d.status_code != 200:
                continue
            hrel = None
            for sen in d.json().get('sensors', []):
                for x in sen.get('data', []):
                    # buscar humedad relativa: hum o hum_out
                    hrel = x.get('hum') or x.get('hum_out')
                    if hrel is not None:
                        break
                if hrel is not None:
                    break
            if hrel is None:
                continue
            nombres.append(n)
            lat.append(la)
            lon.append(lo)
            hum_c.append(hrel)
        return nombres, lat, lon, hum_c
    
    # --- Obtener datos de humedad ---
    nombres, lats, lons, hum_c = obtener_datos_estaciones(station_ids)
    
    # --- Leer shapefile y reproyectar ---
    aca = gpd.read_file("12mun.shp").to_crs(epsg=4326)
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
    z_idw = np.sum(weights * np.take(hum_c, idx), axis=1) / np.sum(weights, axis=1)
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
    zi_masked = np.ma.array(zi, mask=~mask)
    im = ax.imshow(
        zi_masked,
        extent=(minx, maxx, miny, maxy),
        origin='lower',
        cmap="turbo_r",
        alpha=1,
        zorder=1,
        vmin=minimo,
        vmax=maximo
    )
    
    cbar = plt.colorbar(
        im, ax=ax,
        orientation='horizontal',
        fraction=0.02,
        pad=0.04,
        shrink=1,
        aspect=49.5,
        extend='both',
        extendfrac=0.06
    )
    
    # Asegurarnos de incluir 0 a 100 y algunos intermedios
    num_ticks = 6
    ticks = np.linspace(minimo, maximo, num=num_ticks)
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([f"{int(round(t))}%" for t in ticks])
    
    # agregamos las etiquetas para las localidades (igual que antes)
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
    
    # --- Líneas de igual humedad sobre toda la cuadrícula ---
    niveles = np.linspace(0, 100, 10)
    cont_lines = ax.contour(
        xi, yi, zi,
        levels=niveles, colors='black', linewidths=1, zorder=4
    )
    ax.clabel(cont_lines, inline=True, fontsize=8, fmt="%.0f%%")
    
    # Obtener límites actuales del eje (para la comprobación que pediste)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # --- Identificar la humedad máxima ---
    hum_max = max(hum_c) if len(hum_c) > 0 else None
    
    for lon, lat, hum in zip(lons, lats, hum_c):
        # Solo dibujar si el punto está dentro del área visible
        if not (x_min <= lon <= x_max and y_min <= lat <= y_max):
            continue
    
        # Si es la estación con la humedad más alta → fondo destacado
        if hum_max is not None and hum == hum_max:
            facecolor = 'black'
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
            f"{hum:.0f}%",
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
        "   Distribución Espacial de la Humedad Relativa Instantánea\n",
        fontsize=17,
        loc='left'
    )
    
    # validez de los datos (misma lógica)
    dias_semana = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
             "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    ahora = datetime.now()- timedelta(hours=6)
    minuto_redondeado = (ahora.minute // 15) * 15
    fecha_redondeada = ahora.replace(minute=minuto_redondeado, second=0, microsecond=0)
    dia_semana = dias_semana[fecha_redondeada.weekday()]
    dia = fecha_redondeada.day
    mes = meses[fecha_redondeada.month - 1]
    hora_12 = fecha_redondeada.strftime("%I:%M")
    am_pm = "a.m." if fecha_redondeada.strftime("%p") == "AM" else "p.m."
    
    ax.text(
        0, 1.01,
        "Válido el " + dia_semana + " " + str(dia) + " de " + mes + " a las " + hora_12 + " " + am_pm,
        fontsize=11,
        ha='left',
        transform=ax.transAxes,
    )
    
    ax.text(
        1, 1.01,
        "Fuente de datos: SIATM-ACA; SIAT-GRO.",
        fontsize=11,
        ha='right',
        transform=ax.transAxes,
    )
    
    ax.text(
        1, 0.97,
        "Superficie generada mediante interpolación IDW",
        fontsize=11,
        ha='right',
        transform=ax.transAxes,
    )
    
    # Logos (igual que antes)
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
except:
    pass




