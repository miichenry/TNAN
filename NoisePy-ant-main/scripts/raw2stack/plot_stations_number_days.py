# WITH CARTOPY

import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
%matplotlib inline

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Create a background instance.
""" URL for Webserver: https://server.arcgisonline.com/ArcGIS/rest/services/
Available options:
# NatGeo_World_Map (MapServer)
# USA_Topo_Maps (MapServer)
# World_Imagery (MapServer)
# World_Physical_Map (MapServer)
# World_Shaded_Relief (MapServer)
# World_Street_Map (MapServer)
# World_Terrain_Base (MapServer)
# World_Topo_Map (MapServer)
"""
url = 'https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}.jpg'
tile = cimgt.GoogleTiles(url=url)

fig = plt.figure(figsize=(10,10))

# Create a GeoAxes in the tile's projection.
ax = fig.add_subplot(1, 1, 1, projection=tile.crs)
gl = ax.gridlines(draw_labels=True, alpha=0.2)
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# Limit the extent of the map to a small longitude/latitude range. [8.02, 8.44, 47.3, 47.7]
# Aargau: [8, 8.45, 47.35, 47.65]
min_lat = 47.35
max_lat = 47.65
min_lon = 8
max_lon = 8.45

ax.set_extent( [min_lon, max_lon, min_lat, max_lat], crs=ccrs.Geodetic()) # Riehen[7.5, 7.82, 47.486, 47.68]

# Add the background data at zoom level 1.
ax.add_image(tile, 11, interpolation='spline36')

# Use the cartopy interface to create a matplotlib transform object
# for the Geodetic coordinate system. We will use this along with
# matplotlib's offset_copy function to define a coordinate system which
# translates the text by 25 pixels to the left.
geodetic_transform = ccrs.Geodetic()._as_mpl_transform(ax)
text_transform = offset_copy(geodetic_transform, units='dots', x=-25)


# Now add stations
import pandas as pd
stations = pd.read_csv("/home/users/s/savardg/aargau_ant/text_files/station_locations_days.csv")
stations.num_days[stations.num_days == 0 ] = None
lats = stations.latitude.values
longs = stations.longitude.values
ndays = stations.num_days.values
stanames = stations.station.values

#colormap
# import matplotlib.colors as mcol
colormap = plt.get_cmap("jet")
colormap.set_bad("white")

# Scatter
scat = ax.scatter(longs, lats, c=ndays,  cmap=colormap, s=70, alpha=1.0, edgecolor='k', transform=ccrs.Geodetic())
# scat.set_clim(14, 32) # adjust colormap limits

#scat = ax.scatter(longs, lats, c="k",  cmap=colormap, s=70, alpha=1.0, edgecolor='k',
#                  transform=ccrs.Geodetic(), zorder=100)

# # Colorbar
# #get size and extent of axes:
axpos = ax.get_position()
pos_x = axpos.x0+axpos.width + 0.01# + 0.25*axpos.width
pos_y = axpos.y0
cax_width = 0.04
cax_height = axpos.height
#create new axes where the colorbar should go.
#it should be next to the original axes and have the same height!
pos_cax = fig.add_axes([pos_x,pos_y,cax_width,cax_height])
cbar = plt.colorbar(scat, cax=pos_cax)
cbar.set_label('# days of data', rotation=90, fontsize=14)

plt.show()
plt.close()
