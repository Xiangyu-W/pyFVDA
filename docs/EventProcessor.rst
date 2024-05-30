pyFVDA - EventProcessor
=======================

EventProcessor Class Documentation
----------------------------------

Overview
========
The `EventProcessor` class is designed to handle and analyze Solar Energetic Electron (SEE) events from a given list. It provides various methods for loading, creating, and processing event data, including generating figures, performing FVDA analysis, and exporting results.

Class Methods
=============

`__init__(self, event_list, idx)`
--------------------------------
Initializes the `EventProcessor` instance.

**Parameters:**
- `event_list`: DataFrame
  - A list of events with details such as start time, end time, etc.
- `idx`: int
  - Index of the event to be processed.

**Prints:**
- A message indicating the selected event based on the provided index.

`load_fromPickle(file_path)`
----------------------------
Deserializes and loads an `EventProcessor` instance from a pickle file.

**Parameters:**
- `file_path`: str
  - The file path from which to load the pickle file.

**Returns:**
- `EventProcessor`: The instance loaded from the pickle file.

`createEventObject(self, plotfig=True, savefig=False, figpath=None)`
-------------------------------------------------------------------
Creates and configures the CDF object for the event.

**Parameters:**
- `plotfig`: bool (default `True`)
  - Whether to plot figures.
- `savefig`: bool (default `False`)
  - Whether to save figures.
- `figpath`: str (default `None`)
  - Path where figures will be saved. If `None`, a default path is created.

**Returns:**
- `SEEevent`: The configured event object.

`getSolarWindSpeed(self, t_vsw=[])`
-----------------------------------
Performs the solar wind analysis on the event data.

**Parameters:**
- `t_vsw`: list (default `[]`)
  - List of time intervals for solar wind speed analysis.

`prepare_data(self, initial_WindowSize=[7]*6, polyorder=[2]*6, timeForFvda=[])`
-------------------------------------------------------------------------------
Smoothes the data for analysis.

**Parameters:**
- `initial_WindowSize`: list (default `[7]*6`)
  - Window sizes for initial smoothing.
- `polyorder`: list (default `[2]*6`)
  - Polynomial orders for smoothing.
- `timeForFvda`: list (default `[]`)
  - Time intervals for FVDA.

**Returns:**
- `plot_button`: An interactive interface button for plotting.

`getTime_eta(self, bgRange, peakRange, binsForFvda, eta, specialBg={})`
-----------------------------------------------------------------------
Gets the time (eta) for the event.

**Parameters:**
- `bgRange`: list
  - Background range times.
- `peakRange`: list
  - Peak range times.
- `binsForFvda`: list
  - Bins for FVDA.
- `eta`: array
  - Array of eta values.
- `specialBg`: dict (default `{}`)
  - Special background settings.

`getOnsetTime(self, polyorder_toFitOnsetTime=[1]*6)`
---------------------------------------------------
Gets the onset time for the event.

**Parameters:**
- `polyorder_toFitOnsetTime`: list (default `[1]*6`)
  - Polynomial orders for fitting onset time.

`display_figpath(self)`
-----------------------
Displays the figure path where results are stored.

`fvda_approach1(self, eta_forDisplay=[0.7, 0.4, 0.15, 0], etaRangeForFitting=np.arange(0.7,0.1,-0.05))`
--------------------------------------------------------------------------------------------------------
Performs FVDA approach 1.

**Parameters:**
- `eta_forDisplay`: list (default `[0.7, 0.4, 0.15, 0]`)
  - Eta values for display in the deltaTime - 1/beta fitting plot.
- `etaRangeForFitting`: array (default `np.arange(0.7, 0.1, -0.05)`)
  - Eta values for fitting Pathlength and Tr.

`fvda_approach2(self, eta='0', pathLength_forApproach2=[], plotExtraData=None)`
-------------------------------------------------------------------------------
Performs FVDA approach 2.

**Parameters:**
- `eta`: str (default `'0'`)
  - Eta value for approach 2.
- `pathLength_forApproach2`: list (default `[]`)
  - Path lengths for approach 2.
- `plotExtraData`: dict (default `None`)
  - Extra data to be plotted in the same figure.

`exportResultToExcel(self, resultFilePath=None)`
-----------------------------------------------
Exports various attributes of self.event to an Excel file with multiple sheets.

**Parameters:**
- `resultFilePath`: str (default `None`)
  - File path where the Excel file will be saved.

`saveResultToPickle(self, resultFilePath=None)`
-----------------------------------------------
Serializes and saves the current instance to a pickle file.

**Parameters:**
- `resultFilePath`: str (default `None`)
  - File path where the pickle file will be saved.

SEEevent Class Documentation
----------------------------

Overview
========
The `SEEevent` class handles operations on CDF files, including fetching data, filtering, and analysis. It supports different spacecraft data and provides methods for various analyses related to solar wind and electron events.

Class Attributes
================
- `bin_dict`: dict
  - Maps spectral bins to their corresponding IDs.
- `au`: float
  - Astronomical unit in kilometers.

Class Methods
=============

`__init__(self, NAME, direc, t_range, sc='WIND')`
------------------------------------------------
Initializes the `SEEevent` object.

**Parameters:**
- `NAME`: str
  - Name of the event.
- `direc`: list
  - Direction ID for data retrieval.
- `t_range`: list
  - Time range for data query [start, end].
- `sc`: str (default `'WIND'`)
  - Spacecraft identifier.

`fetch_data(self)`
-----------------
Fetches data based on spacecraft and configures data attributes.

`fetch_sta_stb_data(self, cdas)`
-------------------------------
Fetches data for STA or STB spacecraft.

**Parameters:**
- `cdas`: CdasWs
  - CDAS Web Service instance for data retrieval.

`fetch_wind_data(self, cdas)`
-----------------------------
Fetches data for WIND spacecraft.

**Parameters:**
- `cdas`: CdasWs
  - CDAS Web Service instance for data retrieval.

`solarWind(self, t_vsw=[])`
---------------------------
Gets solar wind speed using CDAS Web Service.

**Parameters:**
- `t_vsw`: list (default `[]`)
  - Time range for VSW data.

`fetch_vsw_data(self, sc, vsw_end, t_vsw)`
------------------------------------------
Fetches solar wind data from CDAS.

**Parameters:**
- `sc`: str
  - Spacecraft identifier.
- `vsw_end`: Timestamp
  - End time for data query.
- `t_vsw`: list
  - Time range for VSW data.

**Returns:**
- `DataFrame`: DataFrame with solar wind data.

`plot_vsw_data(self, vsw_data, t_vsw)`
-------------------------------------
Plots the solar wind data after removing outliers.

**Parameters:**
- `vsw_data`: DataFrame
  - DataFrame containing the solar wind data.
- `t_vsw`: list
  - Time range for highlighting specific data.

`calcParker(self, vsw)`
-----------------------
Calculates the Parker spiral length.

**Parameters:**
- `vsw`: float
  - Solar wind speed.

**Returns:**
- float: Parker spiral length.

`peek(self)`
------------
Quickly plots the event data for a visual overview.

`Filter(self, initial_WindowSize=[7]*6, polyorder=[2]*6, timeForFvda=[])`
------------------------------------------------------------------------
Smoothes the data for analysis.

**Parameters:**
- `initial_WindowSize`: list (default `[7]*6`)
  - Initial window sizes for smoothing.
- `polyorder`: list (default `[2]*6`)
  - Polynomial orders for smoothing.
- `timeForFvda`: list (default `[]`)
  - Time range for FVDA analysis.

**Returns:**
- `Button`: An interactive button for plotting.

`setParameters_for_getTime_eta(self, bgRange, peakRange, binsForFvda=[1,2,3,4,5,6], eta=np.array([np.arange(0.7, 0.1, -0.05)]).T, specialBg={})`
------------------------------------------------------------------------------------------------------------------------------------------------
Sets parameters for the `getTime_eta` method.

**Parameters:**
- `bgRange`: list
  - Background range times.
- `peakRange`: list
  - Peak range times.
- `binsForFvda`: list (default `[1,2,3,4,5,6]`)
  - Bins for FVDA.
- `eta`: array (default `np.array([np.arange(0.7, 0.1, -0.05)]).T`)
  - Eta values.
- `specialBg`: dict (default `{}`)
  - Special background settings.

`getT_Eta(self)`
---------------
Calculates the time (eta) for the event using FVDA approach 2.

`calculate_electronBeta(self, energy=[])`
-----------------------------------------
Calculates 1/beta using energy values.

**Parameters:**
- `energy`: list (default `[]`)
  - Energy values in keV.

`calculate_pathLength(self)`
----------------------------
Calculates path length using 1/beta values.

**Returns:**
- `DataFrame`: Path length (`pl`) and release time (`tr`) with their uncertainties.
- `DataFrame`: Time differences (`dt`) for different energy channels.
- `dict`: Fitting parameters for the path length calculation.

`plotFittingResult_pathLength_eta(self, eta=[0.5, 0.4, 0.3, 0.2])`
----------------------------------------------------------------
Plots the path length against 1/beta for the given eta values.

**Parameters:**
- `eta`: list (default `[0.5, 0.4, 0.3, 0.2]`)
  - Eta values for plotting.

`setup_figure(self)`
-------------------
Sets up the figure for plotting intensity profiles and path lengths.

**Returns:**
- `Figure`: Matplotlib figure.
- `Axes`: Matplotlib axes.

`plot_intensityProfile(self, ax, etaForDisplay_pathLength)`
----------------------------------------------------------
Plots the intensity profiles for the given eta values.

**Parameters:**
- `ax`: Axes
  - Matplotlib axes for plotting.
- `etaForDisplay_pathLength`: list
  - Eta values for displaying in the path length fitting plot.

`plot_fitted_path_lengths(self, ax, etaForDisplay_pathLength)`
-------------------------------------------------------------
Plots the fitted path lengths for the given eta values.

**Parameters:**
- `ax`: Axes
  - Matplotlib axes for plotting.
- `etaForDisplay_pathLength`: list
  - Eta values for displaying in the path length fitting plot.

`plot_pathLength_Tr_eta(self, ax, etaRangeForFitting)`
-----------------------------------------------------
Plots the path length and release time against eta values.

**Parameters:**
- `ax`: Axes
  - Matplotlib axes for plotting.
- `etaRangeForFitting`: list
  - Eta values for fitting.

`plot_summary_results_approach1(self, etaForDisplay_pathLength, etaRangeForFitting)`
-----------------------------------------------------------------------------------
Plots the summary results for FVDA approach 1.

**Parameters:**
- `etaForDisplay_pathLength`: list
  - Eta values for displaying in the path length fitting plot.
- `etaRangeForFitting`: list
  - Eta values for fitting.

`plot_FVDA_approach1(self, etaForDisplay_pathLength, etaRangeForFitting=[0.7, 0.4, 0.3, 0.15])`
------------------------------------------------------------------------------------------------
Plots the results for FVDA approach 1.

**Parameters:**
- `etaForDisplay_pathLength`: list
  - Eta values for displaying in the path length fitting plot.
- `etaRangeForFitting`: list (default `[0.7, 0.4, 0.3, 0.15]`)
  - Eta values for fitting.

FVDA Approach 2
---------------

`calculateT0(self, interSect, yita, df, E)`
------------------------------------------
Calculates the onset time T0 for electron events using different polynomial fits.

**Parameters:**
- `interSect`: list
  - Intersection points of eta lines and data.
- `yita`: array
  - Eta values.
- `df`: DataFrame
  - DataFrame with event data.
- `E`: str
  - Energy channel.

**Returns:**
- `DataFrame`: T0 values for different polynomial fits.
- `DataFrame`: Uncertainty in T0 values.

`get_onsetTime(self)`
---------------------
Calculates and sets the onset time for the event.

`calculate_releaseTime(self, eta='0', L=[])`
-------------------------------------------
Calculates electron release times based on path lengths.

**Parameters:**
- `eta`: str (default `'0'`)
  - Eta value.
- `L`: list (default `[]`)
  - Path lengths.

`plot_releaseTimes(self, eta='0')`
----------------------------------
Plots the release times along with annotations and error bars.

**Parameters:**
- `eta`: str (default `'0'`)
  - Eta value.

**Returns:**
- `Figure`: Matplotlib figure with the plot.
- `Axes`: Matplotlib axes with the plot.

`export_toExcel(self, resultFilePath=None)`
------------------------------------------
Exports various attributes of the event to an Excel file with multiple sheets.

**Parameters:**
- `resultFilePath`: str (default `None`)
  - File path where the Excel file will be saved.
