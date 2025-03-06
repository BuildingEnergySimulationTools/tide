import datetime as dt
import math

import numpy as np
import pandas as pd
import requests

from tide.math import cosd, sind

OIKOLAB_PARAM_MAP = {
    "temperature": "temperature (degC)",
    "dewpoint_temperature": "dewpoint_temperature (degC)",
    "mean_sea_level_pressure": "mean_sea_level_pressure (Pa)",
    "wind_speed": "wind_speed (m/s)",
    "100m_wind_speed": "100m_wind_speed (m/s)",
    "relative_humidity": "relative_humidity (0-1)",
    "surface_solar_radiation": "surface_solar_radiation (W/m^2)",
    "direct_normal_solar_radiation": "direct_normal_solar_radiation (W/m^2)",
    "surface_diffuse_solar_radiation": "surface_diffuse_solar_radiation (W/m^2)",
    "surface_thermal_radiation": "surface_thermal_radiation (W/m^2)",
    "total_cloud_cover": "total_cloud_cover (0-1)",
    "total_precipitation": "total_precipitation (mm of water equivalent)",
    "wind_direction": "wind_direction (deg)",
}


def get_oikolab_df(
    lat: float,
    lon: float,
    start: pd.Timestamp | dt.datetime,
    end: pd.Timestamp | dt.datetime,
    api_key: str,
    param: list[str] = None,
    model: str = "era5",
    freq: str = "H",
) -> pd.DataFrame:
    """
    Retrieve weather data from the Oikolab API and return it as a pandas DataFrame.

    This function sends a request to the Oikolab weather API, fetches the specified
    weather parameters for a given location and time range, and returns the data
    in a pandas DataFRame.

    Parameters:
    -----------
    lat : float
        Latitude of the location.
    lon : float
        Longitude of the location.
    start : pd.Timestamp | dt.datetime
        Start date and time for the data request.
    end : pd.Timestamp | dt.datetime
        End date and time for the data request.
    api_key : str
        API key for authentication with the Oikolab service.
    param : list[str], optional
        List of weather parameters to retrieve. If None, the following parameters
        will be fetched. Default is None.
        ['temperature', 'dewpoint_temperature', 'mean_sea_level_pressure',
        'wind_speed', '100m_wind_speed', 'relative_humidity',
        'surface_solar_radiation', 'direct_normal_solar_radiation',
        'surface_diffuse_solar_radiation', 'surface_thermal_radiation',
        'total_cloud_cover', 'total_precipitation']
    model : str, optional
        Weather model to use for data retrieval. Default is "era5".
    freq : str, optional
        Frequency of the data points. Default is "H" (hourly).

    See Oikolab API doc for further informations : https://docs.oikolab.com/references/
    """

    param = list(OIKOLAB_PARAM_MAP.keys()) if param is None else param

    r = requests.get(
        url="https://api.oikolab.com/weather",
        params={
            "param": param,
            "lat": lat,
            "lon": lon,
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "model": model,
            "freq": freq,
            "format": "csv",
        },
        headers={"api-key": api_key},
    )

    if not r.status_code == 200:
        raise ValueError(f"Invalid request. Code:{r.status_code}")

    df = pd.read_csv(r.url, parse_dates=True, index_col=0)
    df.index = df.index.tz_localize("UTC")
    df.index.freq = df.index.inferred_freq
    return df.rename(columns={OIKOLAB_PARAM_MAP[par]: par for par in param})


def sun_position(date: dt.datetime, lat: float = 46.5, long: float = 6.5):
    """
    Returns sun elevation and azimuth angle (Degree) based on latitude and longitude.
    based on Jérôme's python code interpretation of stackoverflow answer to :
    https://stackoverflow.com/questions/8708048/
    position-of-the-sun-given-time-of-day-latitude-and-longitude

    references :
    Michalsky, J.J. 1988. The Astronomical Almanac's algorithm for approximate solar
    position (1950-2050). Solar Energy. 40(3):227-235.

    Michalsky, J.J. 1989. Errata. Solar Energy. 43(5):323.

    Spencer, J.W. 1989. Comments on "The Astronomical Almanac's Algorithm for
    Approximate Solar Position (1950-2050)". Solar Energy. 42(4):353.

    Walraven, R. 1978. Calculating the position of the sun. Solar Energy. 20:393-397.

    :param date: datetime object
    :param lat: latitude in degree
    :param long: longitude in degree
    :return: (elevation, azimuth) in degree
    """

    # Latitude [rad]
    lat_rad = math.radians(lat)

    # Get Julian date - 2400000
    year_begins = dt.datetime(
        year=date.year, month=1, day=1, hour=0, minute=0, second=0, tzinfo=date.tzinfo
    )
    day = (date.replace(hour=0, minute=0, second=0) - year_begins).days

    hour = date.hour + date.minute / 60.0 + date.second / 3600.0
    delta = date.year - 1949
    leap = delta / 4
    jd = 32916.5 + delta * 365 + leap + day + hour / 24

    # The input to the Atronomer's almanach is the difference between
    # The Julian date and JD 2451545.0 (noon, 1 January 2000)
    t = jd - 51545

    # Ecliptic coordinates

    # Mean longitude
    mnlong_deg = (280.460 + 0.9856474 * t) % 360

    # Mean anomaly
    mnanom_rad = math.radians((357.528 + 0.9856003 * t) % 360)

    # Ecliptic longitude and obliquity of ecliptic
    eclong = math.radians(
        (mnlong_deg + 1.915 * math.sin(mnanom_rad) + 0.020 * math.sin(2 * mnanom_rad))
        % 360
    )
    oblqec_rad = math.radians(23.439 - 0.0000004 * t)

    # Celestial coordinates
    # Right ascension and declination
    num = math.cos(oblqec_rad) * math.sin(eclong)
    den = math.cos(eclong)
    ra_rad = math.atan(num / den)
    if den < 0:
        ra_rad = ra_rad + math.pi
    elif num < 0:
        ra_rad = ra_rad + 2 * math.pi
    dec_rad = math.asin(math.sin(oblqec_rad) * math.sin(eclong))

    # Local coordinates
    # Greenwich mean sidereal time
    gmst = (6.697375 + 0.0657098242 * t + hour) % 24
    # Local mean sidereal time
    lmst = (gmst + long / 15) % 24
    lmst_rad = math.radians(15 * lmst)

    # Hour angle (rad)
    ha_rad = (lmst_rad - ra_rad) % (2 * math.pi)

    # Elevation
    el_rad = math.asin(
        math.sin(dec_rad) * math.sin(lat_rad)
        + math.cos(dec_rad) * math.cos(lat_rad) * math.cos(ha_rad)
    )

    # Azimuth
    az_rad = math.asin(-math.cos(dec_rad) * math.sin(ha_rad) / math.cos(el_rad))

    if math.sin(dec_rad) - math.sin(el_rad) * math.sin(lat_rad) < 0:
        az_rad = math.pi - az_rad
    elif math.sin(az_rad) < 0:
        az_rad += 2 * math.pi

    return np.rad2deg(el_rad), np.rad2deg(az_rad)


def aoi_projection(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth):
    """
    https://pvlib-python.readthedocs.io/en/stable/

    Calculates the dot product of the sun position unit vector and the surface
    normal unit vector; in other words, the cosine of the angle of incidence.

    Usage note: When the sun is behind the surface the value returned is
    negative.  For many uses negative values must be set to zero.

    Input all angles in degrees.

    Parameters
    ----------
    surface_tilt : numeric
        Panel tilt from horizontal.
    surface_azimuth : numeric
        Panel azimuth from north.
    solar_zenith : numeric
        Solar zenith angle.
    solar_azimuth : numeric
        Solar azimuth angle.

    Returns
    -------
    projection : numeric
        Dot product of panel normal and solar angle.
    """

    projection = cosd(surface_tilt) * cosd(solar_zenith) + sind(surface_tilt) * sind(
        solar_zenith
    ) * cosd(solar_azimuth - surface_azimuth)

    # GH 1185
    projection = np.clip(projection, -1, 1)

    # GH 1185
    return np.clip(projection, -1, 1)


def beam_component(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, dni):
    """
    === Function extracted from pvlib module ===
    https://pvlib-python.readthedocs.io/en/stable/

    Calculates the beam component of the plane of array irradiance.

    Parameters
    ----------
    surface_tilt : numeric
        Panel tilt from horizontal.
    surface_azimuth : numericbl
        Panel azimuth from north.
    solar_zenith : numeric
        Solar zenith angle.
    solar_azimuth : numeric
        Solar azimuth angle.
    dni : numeric
        Direct Normal Irradiance

    Returns
    -------
    beam : numeric
        Beam component
    """
    beam = dni * aoi_projection(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth
    )
    return np.maximum(beam, 0)


def sky_diffuse(surface_tilt, dhi):
    """
    https://pvlib-python.readthedocs.io/en/stable/

    Determine diffuse irradiance from the sky on a tilted surface using
    the isotropic sky model.

    .. math::

       I_{d} = DHI \frac{1 + \cos\beta}{2}

    Hottel and Woertz's model treats the sky as a uniform source of
    diffuse irradiance. Thus, the diffuse irradiance from the sky (ground
    reflected irradiance is not included in this algorithm) on a tilted
    surface can be found from the diffuse horizontal irradiance and the
    tilt angle of the surface. A discussion of the origin of the
    isotropic model can be found in [2]_.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angle in decimal degrees. Tilt must be >=0 and
        <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    dhi : numeric
        Diffuse horizontal irradiance in W/m^2. DHI must be >=0.

    Returns
    -------
    diffuse : numeric
        The sky diffuse component of the solar radiation.

    References
    ----------
    .. [1] Loutzenhiser P.G. et al. "Empirical validation of models to
       compute solar irradiance on inclined surfaces for building energy
       simulation" 2007, Solar Energy vol. 81. pp. 254-267
       :doi:`10.1016/j.solener.2006.03.009`

    .. [2] Kamphuis, N.R. et al. "Perspectives on the origin, derivation,
       meaning, and significance of the isotropic sky model" 2020, Solar
       Energy vol. 201. pp. 8-12
       :doi:`10.1016/j.solener.2020.02.067`
    """
    return dhi * (1 + cosd(surface_tilt)) * 0.5


def ground_diffuse(surface_tilt, ghi, albedo=0.25):
    """
    https://pvlib-python.readthedocs.io/en/stable/

    Estimate diffuse irradiance on a tilted surface from ground reflections.

    Ground diffuse irradiance is calculated as

    .. math::

       G_{ground} = GHI \times \rho \times \frac{1 - \cos\beta}{2}

    where :math:`\rho` is ``albedo`` and :math:`\beta` is ``surface_tilt``.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angles in decimal degrees. Tilt must be >=0 and
        <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90).

    ghi : numeric
        Global horizontal irradiance. :math:`W/m^2`

    albedo : numeric, default 0.25
        Ground reflectance, typically 0.1-0.4 for surfaces on Earth
        (land), may increase over snow, ice, etc. May also be known as
        the reflection coefficient. Must be >=0 and <=1.

    Returns
    -------
    grounddiffuse : numeric
        Ground reflected irradiance. :math:`W/m^2`

    Notes
    -----
    Table of albedo values by ``surface_type`` are from [2]_, [3]_, [4]_;
    see :py:data:`~pvlib.irradiance.SURFACE_ALBEDOS`.

    References
    ----------
    .. [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
       solar irradiance on inclined surfaces for building energy simulation"
       2007, Solar Energy vol. 81. pp. 254-267.
    .. [2] https://www.pvsyst.com/help/albedo.htm Accessed January, 2024.
    .. [3] http://en.wikipedia.org/wiki/Albedo Accessed January, 2024.
    .. [4] Payne, R. E. "Albedo of the Sea Surface". J. Atmos. Sci., 29,
       pp. 959–970, 1972.
       :doi:`10.1175/1520-0469(1972)029<0959:AOTSS>2.0.CO;2`
    """

    return ghi * albedo * (1 - np.cos(np.radians(surface_tilt))) * 0.5
