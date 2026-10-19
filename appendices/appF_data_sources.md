# Appendix F. Data sources and how to use them in Colab

The chapter examples in this book use synthetic data, so that every number can be reproduced and checked. The labs apply the same methods to real observations. This appendix lists openly available datasets useful for physical hydrology and shows how to reach them from a Google Colab notebook. Dataset versions, access methods and terms change over time: check each provider's current documentation, and always follow its license and citation requirements.

## F.1 Working in Colab

Colab notebooks run on a temporary virtual machine, so a few habits save trouble:

- **Install packages at the top of the notebook**, for example `!pip install cdsapi xarray netCDF4`.
- **Store downloaded data on Google Drive**, mounted with `from google.colab import drive; drive.mount("/content/drive")`, so it survives the end of the session.
- **Keep credentials out of notebooks.** Store API keys in Colab's *Secrets* panel and read them with `from google.colab import userdata; key = userdata.get("CDS_KEY")`. Never paste a key into a cell that will be shared.
- **Request only what you need**: a small region, the needed variables and the needed period. Global, hourly, multi-decade requests can take hours and fill the disk.
- **Use `xarray`** for gridded data in NetCDF or GRIB format; it keeps coordinates and units with the data and makes regional and temporal aggregation simple.

## F.2 Reanalysis

A **reanalysis** combines a weather forecast model with historical observations to produce a consistent, gridded record of the atmosphere and land surface. **ERA5**, produced by ECMWF for the Copernicus Climate Change Service, provides hourly fields from 1940 onward at about 31 km resolution (Hersbach et al., 2020), and **ERA5-Land** provides land variables at about 9 km (Muñoz-Sabater et al., 2021). Variables useful for this book include 2 m temperature and dew point, 10 m wind, surface pressure, radiation fluxes, precipitation, evaporation, snow and soil moisture. Remember that reanalysis precipitation and evaporation are model outputs, constrained only indirectly by observations.

ERA5 is distributed by the Copernicus **Climate Data Store** (CDS, cds.climate.copernicus.eu) under the Copernicus license, which requires attribution. After registering and accepting the dataset's terms, data can be requested with the `cdsapi` package. A request has the following form (check the CDS page of the dataset for the current names and options):

```python
import cdsapi
from google.colab import userdata

c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key=userdata.get("CDS_KEY"))
c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature", "2m_dewpoint_temperature", "surface_solar_radiation_downwards"],
        "year": ["2020"], "month": ["07"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [39, 125, 33, 130],          # north, west, south, east (degrees)
        "data_format": "netcdf",
    },
    "/content/drive/MyDrive/era5_2020_07.nc",
)
```

ERA5 and ERA5-Land are also available in **Google Earth Engine** (for example the collection `ECMWF/ERA5_LAND/DAILY_AGGR`), which is convenient for computing catchment averages without downloading the grids.

## F.3 Satellite gravimetry

The **GRACE** (2002–2017) and **GRACE Follow-On** (2018–) missions measure monthly changes in the Earth's gravity field, from which changes in total terrestrial water storage are derived (Chapter 13). Processed **mascon** solutions from NASA's Jet Propulsion Laboratory and the University of Texas Center for Space Research are the most convenient for hydrology (Watkins et al., 2015). They are distributed through NASA's PO.DAAC and the GRACE project pages (grace.jpl.nasa.gov). Their effective resolution is a few hundred kilometers, and storage changes include groundwater, soil moisture, snow and surface water; separating groundwater requires subtracting the other components using models or observations.

## F.4 Satellite land-surface products

- **Precipitation.** The Integrated Multi-satellitE Retrievals for GPM (**IMERG**) provide half-hourly global estimates from 2000 onward (gpm.nasa.gov). **CHIRPS** combines satellite and gauge data at daily resolution over land between 50° S and 50° N (Funk et al., 2015).
- **Evaporation.** The **MODIS** MOD16 product estimates evaporation with a Penman–Monteith approach (Mu et al., 2011), and **GLEAM** with a Priestley–Taylor approach constrained by soil moisture (Martens et al., 2017). Different products can disagree by 20–30%; compare several (Chapter 8).
- **Soil moisture.** Microwave satellites (SMAP, SMOS) and the merged **ESA CCI** product estimate moisture in the top few centimeters of soil (Dorigo et al., 2017).
- **Vegetation and snow.** MODIS provides leaf area index, vegetation indices, land cover and snow cover at 250–500 m; Landsat and Sentinel-2 provide finer resolution, less frequently.
- **Topography.** Global digital elevation models (SRTM, Copernicus DEM) and hydrologically conditioned products such as **MERIT Hydro** (Yamazaki et al., 2019) supply flow directions and drainage areas for topographic analysis (Chapter 14).

NASA products are distributed through **Earthdata** (earthdata.nasa.gov), which requires free registration; the **AppEEARS** service extracts time series for points and areas without downloading whole tiles. Many of the same products are available in Google Earth Engine.

## F.5 Streamflow and catchment datasets

- **Large-sample catchment datasets** combine streamflow, meteorological forcing and catchment attributes for hundreds to thousands of catchments, and are the best starting point for comparative studies: **CAMELS** for the United States (Addor et al., 2017) and its versions for other countries, and **Caravan**, a global collection in a common format (Kratzert et al., 2023).
- **Global Runoff Data Centre** (GRDC, grdc.bafg.de): daily and monthly discharge from thousands of stations worldwide, provided on request for non-commercial use.
- **National networks.** In the United States, the USGS National Water Information System (waterdata.usgs.gov) provides streamflow and groundwater levels; the `dataretrieval` Python package accesses it directly. In Korea, the Water Resources Management Information System (WAMIS, www.wamis.go.kr) provides water level, discharge, precipitation and evaporation records, and the Korea Meteorological Administration's open data portal (data.kma.go.kr) provides weather observations. Check the terms of use of each portal before redistributing data.
- **Flux networks.** FLUXNET2015 (Pastorello et al., 2020) and regional networks such as AmeriFlux and AsiaFlux provide eddy-covariance measurements of energy, water and carbon fluxes for the labs of Chapters 3 and 8.

## F.6 Good practice

- **Record provenance**: the dataset name, version, access date, region and variables, in the notebook itself.
- **Check units and conventions**: ERA5 accumulated fields (precipitation, radiation, evaporation) are totals over the time step in meters or J m⁻², and evaporation is negative when directed upward.
- **Compare products**: no single dataset is correct; disagreement among independent products is a useful measure of uncertainty.
- **Cite** the datasets you use, following each provider's instructions, in addition to the papers describing them.

## F.7 References

- Addor, N., Newman, A. J., Mizukami, N., and Clark, M. P. (2017). The CAMELS data set: catchment attributes and meteorology for large-sample studies. *Hydrology and Earth System Sciences*, 21, 5293–5313.
- Dorigo, W., et al. (2017). ESA CCI Soil Moisture for improved Earth system understanding: State-of-the art and future directions. *Remote Sensing of Environment*, 203, 185–215.
- Funk, C., et al. (2015). The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes. *Scientific Data*, 2, 150066.
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146, 1999–2049.
- Kratzert, F., et al. (2023). Caravan—A global community dataset for large-sample hydrology. *Scientific Data*, 10, 61.
- Martens, B., et al. (2017). GLEAM v3: satellite-based land evaporation and root-zone soil moisture. *Geoscientific Model Development*, 10, 1903–1925.
- Mu, Q., Zhao, M., and Running, S. W. (2011). Improvements to a MODIS global terrestrial evapotranspiration algorithm. *Remote Sensing of Environment*, 115, 1781–1800.
- Muñoz-Sabater, J., et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13, 4349–4383.
- Pastorello, G., et al. (2020). The FLUXNET2015 dataset and the ONEFlux processing pipeline for eddy covariance data. *Scientific Data*, 7, 225.
- Watkins, M. M., Wiese, D. N., Yuan, D.-N., Boening, C., and Landerer, F. W. (2015). Improved methods for observing Earth's time variable mass distribution with GRACE using spherical cap mascons. *Journal of Geophysical Research: Solid Earth*, 120, 2648–2671.
- Yamazaki, D., Ikeshima, D., Sosa, J., Bates, P. D., Allen, G. H., and Pavelsky, T. M. (2019). MERIT Hydro: a high-resolution global hydrography map based on latest topography dataset. *Water Resources Research*, 55, 5053–5073.

---

*Text: © Daeha Kim, CC BY-NC 4.0.*
