#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from scipy.signal import savgol_filter
from scipy import interpolate
from cdasws import CdasWs
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator
from scipy.optimize import curve_fit
from shapely.geometry import LineString
from matplotlib.widgets import Slider, Button

plt.rc('figure', facecolor='w')  # set savefig background to white



class SEEevent:
    """
    Handles operations on CDF files including fetching data, filtering, and analysis.
    """
    bin_dict = {'Spec_0_E': 35, 'Spec_0_NS': 31, 'Spec_2_E': 37, 'Spec_2_NS': 33}
    au = 1.4959787e8
    
    def __init__(self, NAME, direc, t_range, sc = 'WIND'):
        """
        Initialize the SEEevent object.
        
        Parameters:
        - NAME (str): Name of the event.
        - direc (list): Direction ID for data retrieval.
        - t_range (list): Time range for data query [start, end].
        - sc (str): Spacecraft identifier, defaults to 'WIND'.
        """
        self.NAME = NAME # string
        self.direc = direc
        self.t_range = t_range
        self.onsetHour = -1.
        self.sc = sc
        self.data = []
        self.time = []
        self.plotfig = True
        self.savefig = False
        self.figpath = ''
        self.fetch_data()

        self.vsw_data = None

 
    def fetch_data(self):
        """
        Fetches data based on spacecraft and configures data attributes.
        """
        cdas = CdasWs()
        try:
            if self.sc in ['STA', 'STB']:
                self.fetch_sta_stb_data(cdas)
            elif self.sc == 'WIND':
                self.fetch_wind_data(cdas)
        except Exception as e:
            print(f"Error fetching data: {e}")

        self.energy = np.array([float(i.split('keV')[0]) for i in list(self.data[0].columns)])
        Me = 0.511
        E = Me+self.energy*10**-3
        p = np.sqrt(E**2-Me**2)
        # self.beta = p/E

    def fetch_sta_stb_data(self, cdas):
        """
        Fetch data for STA or STB spacecraft.
        """
        direcName = list(self.bin_dict.keys())[list(self.bin_dict.values()).index(self.direc[0])]
        data = cdas.get_data('%s_L1_SEPT'%self.sc, [direcName], self.t_range[0], self.t_range[1])[1]
        energy = [2, 4, 6, 7, 10, 12] # ['49.7 keV', '69.8 keV', '94.5 keV', '115.0 keV', '179.0 keV', '240.0 keV'] #[1,2,4,6,7,10]#
        colName = [str(i)+' keV' for i in np.around(data['Spec_E_Mean_Energy'][...][energy],decimals=1)]
        self.data.append(pd.DataFrame(data[direcName][...][:, energy], columns=colName)) # list, cdf data is stored in self.data[0]
        self.time = [t.replace(microsecond=0) for t in data['Epoch_E'][...]]

    def fetch_wind_data(self, cdas):
        """
        Fetch data for WIND spacecraft.
        """
        var = f'FLUX_byE_atA_stackPA{self.direc[0]-1}'
        data = cdas.get_data('WI_SFPD_3DP', [var], self.t_range[0], self.t_range[1])[1]
        colName = [str(i)+' keV' for i in [26.60,40.2,66.3,108.6,182.4,310,520]]
        self.data.append(pd.DataFrame(data[var][...][:, 0:7], columns=colName)) # list, cdf data is stored in self.data[0]
        self.time = list(data['Epoch'][...])
        

    def solarWind(self, t_vsw=[]):
        '''get solar wind speed using cda_sws()
        
        t_vsw: list, [start, end], pd.Timestamp. Time range for VSW data.
        '''
        import math as m
        from cdasws import CdasWs
        cdas = CdasWs()

        t_range = self.t_range
        hour = self.onsetHour

        # 计算 vsw 的截止时间（定在electron onsetHour 之前 5min）
        vsw_end = pd.Timestamp('%sT%d:%d:00Z'%(t_range[0][0:10],round(m.modf(hour)[1]),round(m.modf(hour)[0]*60)))-timedelta(minutes=5)
        if not t_vsw: # 如果没有指定时间段，默认为 onsetHour 前 11 小时
            t_vsw=[vsw_end-timedelta(hours=11),vsw_end] # data type: pd.Timestamp

        self.vsw_data = self.fetch_vsw_data(self.sc, vsw_end, t_vsw)
        self.plot_vsw_data(self.vsw_data, t_vsw)


    def fetch_vsw_data(self, sc, vsw_end, t_vsw):
        """
        Fetch solar wind data from CDAS.

        Parameters:
        - sc: Spacecraft identifier.
        - vsw_end: End time for data query.
        - t_vsw: Time range for VSW data.

        Returns:
        - DataFrame with solar wind data.
        """
        sc = sc.upper()
        cdas = CdasWs()
        if sc == 'WIND':
            dataset_id = 'WI_EHPD_3DP'
            variable = 'VSW'

            vsw_raw = cdas.get_data(dataset_id, [variable], 
                                (vsw_end - timedelta(hours=22)).strftime('%Y-%m-%dT%H:%M:%SZ'), 
                                vsw_end.strftime('%Y-%m-%dT%H:%M:%SZ'))[1]
            vsw_data = pd.DataFrame(vsw_raw['VSW'][...],columns=['Vx','Vy','Vz'])
            vsw_data['V'] = vsw_data.apply(lambda x: round(np.sqrt(x['Vx']**2 + x['Vy']**2+ x['Vz']**2),2),axis=1) # solar wind speed vector and scalar

        elif sc == 'STA':
            dataset_id = 'STA_L2_MAGPLASMA_1M'
            variable = 'Vp_RTN'

            vsw_raw = cdas.get_data(dataset_id, [variable], 
                                (vsw_end - timedelta(hours=22)).strftime('%Y-%m-%dT%H:%M:%SZ'), 
                                vsw_end.strftime('%Y-%m-%dT%H:%M:%SZ'))[1]
            vsw_data = pd.DataFrame(vsw_raw['Vp_RTN'][...],columns=['V'])

        vsw_data['Epoch'] = vsw_raw['Epoch'][...]
        vsw_data.set_index('Epoch', inplace=True)
        vsw_data = vsw_data.dropna(axis=0,how='any')
        vsw_data.index = vsw_data.index.tz_localize('utc') # convert naive timestamp to timezone-aware
        return vsw_data

    def plot_vsw_data(self, vsw_data, t_vsw):
        """
        Plot the solar wind data after removing outliers.
        
        Parameters:
        - vsw_data: DataFrame containing the solar wind data.
        - t_vsw: Time range for highlighting specific data.
        """
        # Remove outliers using the IQR method
        Q1 = vsw_data['V'].quantile(0.25)
        Q3 = vsw_data['V'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR
        filtered_data = vsw_data[(vsw_data['V'] >= lower_bound) & (vsw_data['V'] <= upper_bound)]
        # calculate mean VSW
        vsw = filtered_data.loc[t_vsw[0]:t_vsw[1],'V'].mean()
        std = filtered_data.loc[t_vsw[0]:t_vsw[1],'V'].std()

        fig, ax = plt.subplots()
        filtered_data.plot(ax=ax, y='V', style='--', color='#000957', label='Solar Wind Speed', linewidth=1.5)
        filtered_data.loc[t_vsw[0]:t_vsw[1]].plot(ax=ax, y='V', color='#7C83FD', linewidth=1.8, label='_')
        
        ax.set_xlabel(t_vsw[0].strftime('%Y-%m-%d'), fontsize=16)
        ax.set_ylabel('Plasma Speed (km/s)', fontsize=16)
        ax.legend(frameon=False, fontsize=12)
        ax.text(0.65,0.15, '$\overline{V}$ = %.1f km/s'%(vsw), color='#7C83FD', fontsize = 12, transform=ax.transAxes)
        ax.text(0.65,0.08, '$\sigma$ = %.1f km/s'%(std), color='#7C83FD', fontsize = 12, transform=ax.transAxes) #  #FF5151
        
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(HourLocator(interval=3))
        ax.margins(x=0)
        plt.xticks(fontsize=10, rotation=0, ha='center')
        plt.yticks(fontsize=10)
        plt.subplots_adjust(bottom=0.15, top=0.9)
        
        if self.savefig:
            fig.savefig(self.figpath + f'VSW_{self.NAME}.png', dpi=300)
        if self.plotfig:
            plt.show()

        # Calculate Parker field length
        self.vsw = filtered_data['V'].mean()
        self.parker = self.calcParker(self.vsw)

    
    def calcParker(self, vsw):
        import math
        Sun=6.955e5
        omega=2.86e-6 # rad/s
        
#         self.vsw = float(input('Enter the solar wind speed(km/s):'))
        self.vsw = vsw
        b=self.vsw/omega
        radius = 1.0
        r_p=radius*SEEevent.au
        return (0.5*r_p*np.sqrt(1+(r_p/b)**2)+0.5*b*math.log(r_p/b+np.sqrt(1+(r_p/b)**2)))/SEEevent.au
        
    def peek(self):
        fig = plt.figure(); ax1 = fig.add_subplot(1,1,1)
        temp = self.data[0].join(pd.DataFrame(self.time, columns = ['Date']))
        temp.set_index('Date').plot(ax = ax1)
        
        ax1.set_xlabel('Time (UTC)', fontsize = 15)
        ax1.set_yscale('log')
        ax1.autoscale(enable=True, axis='x', tight=True)
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))

        if self.savefig == True:
            fig.savefig(self.figpath + '%s_quick_plot.png'%self.NAME, dpi=300)
            print('Quick plot saved as' + self.figpath + '%s_quick_plot.png'%self.NAME)
        
        if self.plotfig == True:
            plt.show()


    def Filter(self, initial_WindowSize = [7]*6, polyorder = [2]*6, timeForFvda = []):
        
        # Initialize object variables
        # timeForFvda: time range for FVDA analysis
        if not timeForFvda: # if timeForFvda is empty
            self.timeForFvda = [datetime.strptime(i, "%Y-%m-%dT%H:%M:%SZ") for i in self.t_range]
        else:
            self.timeForFvda = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in timeForFvda]
        
        self.smoothWindowSize = initial_WindowSize.copy() # initial window size for savgol_filter
        self.polyorder = polyorder # polynomial order for savgol_filter


        # initialize temp variables
        lineObj = [[0] * 3 for _ in range(2)]
        win_size = [[0] * 3 for _ in range(2)]
        plot_button = [[0] * 3 for _ in range(2)]
        energyBins = list(self.data[0].columns)
        
                
        # set figure parameters
        fig, axes = plt.subplots(2,3, sharex = True)
        fig.set_size_inches(13.5,8) #重新设置大小
        plt.subplots_adjust(hspace=0, wspace = 0.2)

        # rawData: raw data without nan
        rawData = self.data[0].copy(deep=True)
        rawData['time'] = self.time
        rawData.set_index('time', inplace = True)
        rawData = rawData.dropna(axis=0,how='any')
        
#         rawData.loc[pd.Timestamp('2016-07-23 05:46:00'):pd.Timestamp('2016-07-23 06:00:00'),['108.6 keV','182.4 keV','310 keV']] = rawData.loc[pd.Timestamp('2016-07-23 05:46:00'):pd.Timestamp('2016-07-23 06:00:00'),['108.6 keV','182.4 keV','310 keV']]*0.3 # 2016-07-23
        
        self.dataF = rawData.loc[self.timeForFvda[0]:self.timeForFvda[1]].copy(deep=True) # must use deep copy
        
        for i in range(2):
            for j in range(3):
#                 p[i][j], = axes[i,j].plot(self.time, self.data[0][energyBins[i*3+j]], '-o')
#                 yF = savgol_filter(self.data[0][energyBins[i*3+j]][~nanIdx], initial_WindowSize[i*3+j], 3)
#                 p[i][j],= axes[i,j].plot(list(compress(self.time, ~nanIdx)), yF, '-')
                lineObj[i][j], = axes[i,j].plot(rawData.loc[self.timeForFvda[0]:self.timeForFvda[1]].index, rawData.loc[self.timeForFvda[0]:self.timeForFvda[1], energyBins[i*3+j]], '-o')
                
                if initial_WindowSize[i*3+j] != 0:
                    yF = savgol_filter(rawData.loc[self.timeForFvda[0]:self.timeForFvda[1], energyBins[i*3+j]], initial_WindowSize[i*3+j], polyorder[i*3+j]) # 对 self.timeForFvda 内的数据进行平滑
                    lineObj[i][j],= axes[i,j].plot(rawData.loc[self.timeForFvda[0]:self.timeForFvda[1]].index, yF, '-')
                    self.dataF[energyBins[i*3+j]] = yF
#                 axes[i,j].set_xlim(self.timeForFvda[0], self.timeForFvda[1])
                axes[i,j].xaxis.set_major_formatter(DateFormatter('%H:%M'))
                axes[i,j].autoscale(enable=True, axis='x', tight=True)

                # create sliders
                pos = axes[i, j].get_position()
                if i == 0:
                    # Defining the Slider button
                    ax_slide = plt.axes([pos.x0, pos.y1, pos.width, 0.03]) #xposition, yposition, width and height
                    if j == 0:
                        # Properties of the slider
                        win_size[i][j] = Slider(ax_slide, label='Win size', valmin=1, valmax=99, valinit=initial_WindowSize[i*3+j], valstep=2)
                    else:
                        win_size[i][j] = Slider(ax_slide, label='', valmin=1, valmax=99, valinit=initial_WindowSize[i*3+j], valstep=2)
                elif i == 1:
                    # Defining the Slider button
                    ax_slide = plt.axes([pos.x0, pos.y0 - 0.08, pos.width, 0.03]) #xposition, yposition, width and height

                    if j == 0:
                        # Properties of the slider
                        win_size[i][j] = Slider(ax_slide, label='Win size', valmin=1, valmax=99, valinit=initial_WindowSize[i*3+j], valstep=2)
                    else:
                        win_size[i][j] = Slider(ax_slide, label='', valmin=1, valmax=99, valinit=initial_WindowSize[i*3+j], valstep=2)
        
        def draw(val):
            for i in range(2):
                for j in range(3):
#                     if mode == True:
#                     if initial_WindowSize[i*3+j] != 0:
                    current_v = int(win_size[i][j].val)
                    self.dataF[energyBins[i*3+j]] = savgol_filter(rawData.loc[self.timeForFvda[0]:self.timeForFvda[1], energyBins[i*3+j]], current_v, polyorder[i*3+j])
                    lineObj[i][j].set_ydata(self.dataF[energyBins[i*3+j]])
                    axes[i,j].get_figure().canvas.draw()
                    self.smoothWindowSize[i*3+j] = win_size[i][j].val
            
        
        # create buttons
        ax_button = plt.axes([0.15, 0.8, 0.06,0.04])

        #Properties of the button
        plot_button = Button(ax_button, 'Update', color = 'white', hovercolor = 'grey')
        plot_button.on_clicked(draw)
        if self.savefig == True:
            fig.savefig(self.figpath + '%s_Filter.png'%self.NAME, dpi=300)
        plt.show()
#         import pdb; pdb.set_trace() # 断点
        return plot_button


    def setParameters_for_getTime_eta(self, bgRange, peakRange, 
                                      binsForFvda=[1,2,3,4,5,6], 
                                      eta = np.array([np.arange(0.7, 0.1, -0.05)]).T, 
                                      specialBg={}):
        ''' Set parameters for getTime_eta method.'''
        self.binsForFvda = binsForFvda
        self.bgRange = bgRange
        self.peakRange = peakRange
        self.eta = eta
        self.specialBg = specialBg

    def getT_Eta(self):
        # FVDA Approach 2
        # initiate dataframe for FVDA analysis, and onset time T0
        data_FVDA = self.dataF.iloc[:,[i-1 for i in self.binsForFvda]].copy(deep=True)
        self.T0 = pd.DataFrame(columns = data_FVDA.columns, index = ['poly2', 'poly3', 'poly1'])
        self.T0_uncertaintyFromDiffPoly = pd.DataFrame(columns = data_FVDA.columns, index = ['poly2', 'poly3', 'poly1'])

        bg_dict = {} # 包含 background range 之间的数据
        for i in data_FVDA.columns:
            bg_dict[i] = data_FVDA.loc[self.bgRange[0]:self.bgRange[1], i]

        # extract background info at desired energy channels
        MEAN = data_FVDA.loc[self.bgRange[0]:self.bgRange[1]].mean()
        idmax = data_FVDA.loc[self.peakRange[0]:self.peakRange[1]].idxmax() # max id
        # idmax = data_FVDA.loc[self.bgRange[1]:self.timeForFvda[1]].idxmax() # max id, old version
        

        # revise 2022-03-12: use a dict to process bkg. For specified energy channels, use bgRange stored in the dict. Others use default bgRange.
        if self.specialBg:
            for i in self.specialBg.keys():
                # replace mean and max index @ specified energy channel
                MEAN[i] = data_FVDA.loc[self.specialBg[i][0]:self.specialBg[i][1],i].mean()
                idmax[i] = data_FVDA.loc[self.specialBg[i][1]:self.peakRange[1],i].idxmax()

                bg_dict[i] = data_FVDA.loc[self.specialBg[i][0]:self.specialBg[i][1], i]

        # obtain max value and calculate max-mean    
        maxVal = [data_FVDA[i][idmax[i]] for i in idmax.index] # max value
        max_mean = np.array([list(map(lambda x,y: x-y,maxVal, MEAN))]) # max - mean
        yita = self.eta
        
        # corresponding Y for several yitas
        yYita = pd.DataFrame(yita.dot(max_mean)+np.array(MEAN), columns = data_FVDA.columns)
        # relavant parameters
        self.yYita = yYita
        self.peak = idmax
        self.bg = bg_dict
        

        # initialize onset time DataFrame
        idx = ['%.2f'%i for i in yita]
        tYita = pd.DataFrame(index = pd.Series(idx)); tYita.index.name = 'eta'
        # extract desired time (around rising phase)
        # xEpoch = [i.timestamp() for i in list(data_FVDA[self.bgRange[1]:self.timeForFvda[1]].index)]

        # j: energy channels
        # i: yita
        for j in list(data_FVDA.columns):

            xEpoch = np.array([i.timestamp() for i in list(data_FVDA[self.bg[j].index[-1]:idmax[j]].index)])

            interSect = []
            Line = []
            Line1 = LineString(np.column_stack((np.array(xEpoch), data_FVDA[j][self.bg[j].index[-1]:idmax[j]])))
            # Line1 = LineString(np.column_stack((np.array(xEpoch), data_FVDA[j][self.bgRange[1]:self.timeForFvda[1]])))
            
            for i in np.arange(yita.shape[0]):
                Line.append(LineString([(xEpoch[0], yYita[j][i]),(xEpoch[-1], yYita[j][i])]))
                intersectPoints = Line1.intersection(Line[i])

                # Determine if there are multiple points of intersection. If so, choose the one on the rising phase.
                if intersectPoints.geom_type == 'MultiPoint':
                    desiredPoint = intersectPoints.geoms[0]
                    for pt in intersectPoints.geoms:
                        if pt.x > desiredPoint.x:
                            desiredPoint = pt
                    interSect.append(np.array(desiredPoint.coords))
                else:
                    interSect.append(np.array(intersectPoints.coords))
#             import pdb; pdb.set_trace() # 断点
            
            # convert 2D array to 1D
            for k in np.arange(len(interSect)):
                if interSect[k].ndim != 1:
                    interSect[k] = interSect[k][-1]
            
#   FVDA Approach 2
#             get t(0)
            self.T0[j], self.T0_uncertaintyFromDiffPoly[j] = self.calculateT0(interSect, yita, data_FVDA[j][self.timeForFvda[0]:self.timeForFvda[1]], j)
            
#             import pdb; pdb.set_trace() # 断点
            # timestamp to datetime
            for m in np.arange(len(interSect)):
                if interSect[m].size != 0: # if size = 0, no intersections
                    interSect[m] = np.array([datetime.fromtimestamp(interSect[m][0], tz = timezone.utc),interSect[m][1]])
            tYita[j] = interSect
        
        self.t_eta = tYita
        # return tYita
    
    def calculate_electronBeta(self, energy = []):
        """Calculate 1/beta using energy values.
        
        enegy: list, energy values in keV."""
        
        if not energy:
            energy = np.array([float(i.split('keV')[0]) for i in list(self.onsetTime.columns)])
        else:
            energy = np.array(energy)
        
        Me = 0.511  # Rest mass of electron, MeV
        E = Me + energy * 1e-3  # Convert keV to MeV
        momentum = np.sqrt(E**2 - Me**2)
        self.beta = momentum / E
        # self.oneOverBeta = 1 / (momentum / E)
        # return self.oneOverBeta


#  FVDA Approach 1
    def calculate_pathLength(self):
        """Calculate path length using 1/beta values."""
        
        au = 1.4959787e8

        df = self.onsetTime
        PL = pd.DataFrame(index = df.index, columns = ['pl', 'tr'])
        dt = pd.DataFrame(index = df.index, columns = df.columns)
        opt = {}
        self.calculate_electronBeta() # Calculate electron beta values
        
        def func(x, a, b):
            return a*x+b
        
        for i in df.index:
            y = []
            for j in df.columns:
                y.append( round( df.loc[i,j][0].timestamp() - df.loc[i,df.columns[-1]][0].timestamp(),3 ) )
            
            opt[i], cov = curve_fit(func, 1/self.beta, y)
            PL.loc[i, 'pl'] = [ round(opt[i][0]*3*10**5/au,4), round(np.sqrt(np.diag(cov))[0]*3*10**5/au,4) ] # path length
            PL.loc[i, 'tr'] = [datetime.fromtimestamp(round(opt[i][1]+df.loc[i,df.columns[-1]][0].timestamp(),3), tz = timezone.utc), 
                            round(np.sqrt(np.diag(cov))[1],3)]
            dt.loc[i] = y
        
        return PL, dt, opt

    def plotFittingResult_pathLength_eta(self, eta=[0.5, 0.4, 0.3, 0.2]):
        '''Plot path length -- 1/beta'''
        '''
        self.pathLength_tr: FVDA Approach 1: path length and release time
        self.delta_onsetTime: delta time of electrons of different energy channels
        '''

        def func(x, a, b):
            return a*x+b
        
        self.pathLength_tr, self.delta_arrivalTime, self.fittingParam_PL_tr = self.calculate_pathLength()
        
        # Determine the number of subplots needed based on the length of yita
        num_plots = len(eta)
        fig, axs = plt.subplots(num_plots, 1, figsize=(5, 2 * num_plots), sharex=True)  # Adjust the height based on the number of plots
        fig.subplots_adjust(left=0.15, right=0.95, top=0.95,  bottom=0.1 ,hspace=0.0)

        # If there's only one plot, axs will not be an array, wrap it in a list
        if num_plots == 1:
            axs = [axs]
        
        for i, eta in enumerate(eta):
            idx = '%.2f' % eta if eta!=0 else '0' # Convert float to string
            # Function 'func' needs to be defined or passed to this method
            # 'opt' needs to be accessible here, assuming it's a class attribute or passed as a parameter
            
            # Plot the model and data points
            axs[i].plot(1/self.beta, func(1/self.beta, *self.fittingParam_PL_tr[idx]), color='orange')
            axs[i].plot(1/self.beta, self.delta_arrivalTime.loc[idx], 'o')
            
            # Adding text and setting labels
            axs[i].text(0.05, 0.9, '$L (\eta = %s)$ =%.2f$\pm$%.2f AU'%(idx, self.pathLength_tr.loc[idx,'pl'][0], self.pathLength_tr.loc[idx,'pl'][1]),
                        verticalalignment='top', horizontalalignment='left', transform=axs[i].transAxes, fontsize=11)
            axs[i].set_ylabel('second (s)', fontsize=14)
            axs[i].tick_params(axis='both', which='both', direction='inout')

        # Label the shared x-axis
        axs[-1].set_xlabel(r'1/$\beta$', fontsize=14)
        
        # Saving the figure if required
        if self.savefig:
            fig.savefig(self.figpath + '%s_path length_approach1.png'%self.NAME, dpi=300)

        # Show the plot
        if self.plotfig:
            plt.show()

    # ==================== Plot Summary results of FVDA approach 1 ====================
    # Constants
    MARKER_TYPES = ['o', 'd', 's', 'P','^', 'v']
    COLORS = ['r', 'g', 'b', 'm', 'C9', 'C5']

    # Utility functions
    def setup_figure(self):
        fig, ax = plt.subplots(len(1/self.beta), 2, figsize=(12, 2*len(self.beta)))
        fig.subplots_adjust(left=0.1, right=0.95, top=0.98, bottom=0.08, hspace=0.2)
        plt.rcParams['ytick.labelsize'] = 12
        return fig, ax

    def plot_intensityProfile(self, ax, etaForDisplay_pathLength):

        rawData = self.data[0].copy()
        rawData['time'] = self.time
        rawData.set_index('time', inplace = True)
        rawData = rawData.dropna(axis=0,how='any')

        colName = self.onsetTime.columns.to_list()

        # Plot raw data and filtered data
        for i, j in enumerate(colName):
            ax[i, 0].plot(rawData.loc[self.timeForFvda[0]:self.timeForFvda[1]].index, rawData.loc[self.timeForFvda[0]:self.timeForFvda[1], j],color ='#393E46') # metadata
            ax[i, 0].plot(self.dataF.loc[self.timeForFvda[0]:self.timeForFvda[1]].index, self.dataF.loc[self.timeForFvda[0]:self.timeForFvda[1], j], linewidth = 2, color='#F3A953') # filtered data
            
            # Plot background range
            if j in list(self.specialBg.keys()):
                specialBkg = self.specialBg[colName[i]]
                ax[i, 0].plot(self.dataF.loc[specialBkg[0]:specialBkg[1]].index, self.dataF.loc[specialBkg[0]:specialBkg[1], j], linewidth = 2, color='#366ED8')# bkg
            else:
                ax[i, 0].plot(self.dataF.loc[self.bgRange[0]:self.bgRange[1]].index, self.dataF.loc[self.bgRange[0]:self.bgRange[1], j], linewidth = 2, color='#366ED8')# bkg
            
            ax[i, 0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax[i, 0].autoscale(enable=True, axis='x', tight=True)
            ax[i, 0].tick_params(axis='both', which='both', direction='inout')

            ax[i, 0].set_ylabel(j+'\n$\\regular_{/cm^2/s/sr/MeV}$',fontsize=15)
            ax[i, 0].ticklabel_format(style='sci', axis='y', scilimits=(0.0,0.0)) # y style: sci notation

            if i != len(colName)-1:
                ax[i, 0].set_xticklabels([])
            
            # Plot t(eta) on intensity profiles
            etaForDisplay = etaForDisplay_pathLength[-( len(1/self.beta)-2 ):]
            xPt = [m[0] for m in self.onsetTime.loc[etaForDisplay, self.onsetTime.columns[i]]]
            yPt = [m[1] for m in self.onsetTime.loc[etaForDisplay, self.onsetTime.columns[i]]]
            for xp, yp, m, c in zip(xPt, yPt, SEEevent.MARKER_TYPES, SEEevent.COLORS):
                ax[i, 0].plot(xp, yp, marker = m, color = c, markersize=6.5)
            
            # 1/beta
            ax[i, 0].text(0.05,0.8, r'1/$\beta$=%.1f'%(1/self.beta[i]), fontsize = 15, transform=ax[i, 0].transAxes)

        ax[-1, 0].set_xlabel(self.timeForFvda[0].strftime('%Y-%m-%d %H:%M:%S')[:10], fontsize=17)
        ax[-1, 0].tick_params(axis='x', which='both', labelsize=14)

    def plot_fitted_path_lengths(self, ax, etaForDisplay_pathLength):
        def linear_model(x, a, b):
            return a*x + b

        etaForDisplay = etaForDisplay_pathLength[-( len(1/self.beta)-2 ):]
        pl_tr = self.pathLength_tr

        for idx, eta in enumerate(etaForDisplay):
            dt = self.delta_arrivalTime.loc[eta]
            opt = self.fittingParam_PL_tr

            ax[idx, 1].plot(1/self.beta, linear_model(1/self.beta, *opt[eta]), color = 'orange')
            ax[idx, 1].plot(1/self.beta, dt, SEEevent.MARKER_TYPES[idx], color=SEEevent.COLORS[idx], markersize=6.5, label='$\eta$='+ eta)

            ax[idx, 1].set_ylabel('seconds', fontsize=16)
            ax[idx, 1].text(0.55,0.15, 'L=%.2f$\pm$%.2f AU'%(pl_tr.loc[eta,'pl'][0], pl_tr.loc[eta,'pl'][1]), color=SEEevent.COLORS[idx], fontsize = 10, 
                                    transform=ax[idx, 1].transAxes)
                
            ax[idx, 1].legend(fontsize=12)
            ax[idx, 1].set_ylabel('seconds',fontsize=16)
            ax[idx, 1].tick_params(axis='both', which='both', direction='inout')
            
            #  Will be improved in the future. Number of panels will be flexible.
            if idx != len(etaForDisplay)-1:
                ax[idx, 1].set_xticklabels([])
        
        ax[-3, 1].tick_params(axis='x', which='both', direction='inout',labelsize=12)
        ax[-3, 1].set_xlabel(r'1/$\beta$', fontsize = 12, labelpad=-7)

    def plot_pathLength_Tr_eta(self, ax, etaRangeForFitting):
        def linear_model(x, a, b):
            return a*x + b

        # **************** plot PL/Tr -- eta **************** #
        pl = [i[0] for i in self.pathLength_tr.iloc[:-1, 0]]
        pl_unc = [i[1] for i in self.pathLength_tr.iloc[:-1, 0]]
        tr = [i[0] for i in self.pathLength_tr.iloc[:-1, 1]]
        tr_unc = [timedelta(seconds = i[1]) for i in self.pathLength_tr.iloc[:-1, 1]]

        # Path length - eta, fitting
        etaStr = ['%.2f'%i for i in etaRangeForFitting] # eta in strings
        optPathLen, covPathLen = curve_fit(linear_model, etaRangeForFitting, [i[0] for i in self.pathLength_tr.loc[etaStr, 'pl']])
        # Tr - eta fitting. 相对时间差, second. 
        deltaTimeArray = np.array([i[0].timestamp() for i in self.pathLength_tr.loc[etaStr,'tr']]) - self.pathLength_tr.loc[etaStr[-1], 'tr'][0].timestamp()
        optTr, covTr = curve_fit(linear_model, etaRangeForFitting, deltaTimeArray)

        ax[-2,1].errorbar(self.eta, pl, yerr = pl_unc, marker='_', ls='none', label = '$L(\eta)$') # data points
        ax[-2,1].plot(np.append(etaRangeForFitting,0), linear_model(np.append(etaRangeForFitting,0), *optPathLen), '--') # fit curve
        ax[-2,1].text(0.1,0.05, '%.2f $\pm$ %.2fau'% (optPathLen[1], np.sqrt(np.diag(covPathLen))[1]), color='g',fontsize = 13, transform=ax[-2,1].transAxes)
        ax[-2,1].errorbar(0, optPathLen[1], yerr = np.sqrt(np.diag(covPathLen))[1], marker = 's')
        ax[-2,1].legend()
        
        ax[-2,1].set_ylabel('$L(\eta)$ (au)', fontsize = 14)
        ax[-2,1].set_xticklabels([])
        
        ax[-1,1].errorbar(self.eta, tr, yerr = tr_unc, marker='_', ls='none', label = '$T_r(\eta)$')
        fittedTr_approach1 = [datetime.fromtimestamp(i, tz = timezone.utc) for i in (linear_model(np.append(etaRangeForFitting,0), *optTr)+self.pathLength_tr.loc[etaStr[-1], 'tr'][0].timestamp())]
        ax[-1,1].plot(np.append(etaRangeForFitting,0), fittedTr_approach1, '--')
        ax[-1,1].text(0.1,0.05, fittedTr_approach1[-1].strftime("%H:%M:%S")+' $\pm$ %.2fs'% np.sqrt(np.diag(covTr))[1], color='g',fontsize = 13, transform=ax[-1,1].transAxes)
        ax[-1,1].errorbar(0, fittedTr_approach1[-1], yerr = timedelta(seconds = np.sqrt(np.diag(covTr))[1]), marker = 's')
        ax[-1,1].yaxis.set_major_formatter(DateFormatter('%H:%M'))

        ax[-1,1].set_xlabel('$\eta$',fontsize=16)
        ax[-1,1].set_ylabel('$T_r(\eta)$', fontsize = 14)
        ax[-1,1].tick_params(axis='x', labelsize=14)
        ax[-1,1].legend()

        self.pathLength_tr_approach1 = pd.DataFrame({'pl':[optPathLen[1], np.sqrt(np.diag(covPathLen))[1]], 
                                    'tr':[fittedTr_approach1[-1].strftime("%Y-%m-%d %H:%M:%S"), np.sqrt(np.diag(covTr))[1]]}, index = ['value','uncertainty'])


    def plot_summary_results_approach1(self, etaForDisplay_pathLength, etaRangeForFitting):
        fig, ax = self.setup_figure()

        # ****** Plot intensity profiles - approach 1 *******
        self.plot_intensityProfile(ax, etaForDisplay_pathLength)

        # ****** Plot fitted path lengths - approach 1 *******
        self.plot_fitted_path_lengths(ax, etaForDisplay_pathLength)

        # ****** Plot pathLength-eta, Tr-eta fitting  *******
        self.plot_pathLength_Tr_eta(ax, etaRangeForFitting)
        
        if self.savefig:
            fig.savefig(f'{self.figpath}{self.NAME}_FVDA_Approch1_summary.png', dpi=300)
        if self.plotfig:
            plt.show()

    def plot_FVDA_approach1(self, etaForDisplay_pathLength, etaRangeForFitting=[0.7, 0.4, 0.3, 0.15]):
        self.plot_summary_results_approach1(etaForDisplay_pathLength, etaRangeForFitting)

# ==================== Plot Summary results of FVDA approach 1 ====================

#  FVDA Approach 2
    def calculateT0(self, interSect, yita, df, E):
        T0 = pd.DataFrame(columns = [E], index = ['poly2', 'poly3', 'poly1'])
        T0_uncertainty = pd.DataFrame(columns = [E], index = ['poly2', 'poly3', 'poly1'])

        from copy import deepcopy
        timeSeries = deepcopy(interSect)
        time = [datetime.fromtimestamp(i[0], tz = timezone.utc) for i in timeSeries] # t(yita), type: datetime
        # tYita = [i[0] for i in timeSeries] #  t(yita), type: timestamp
        tYita = [(time[0]-i).total_seconds() for i in time]
        yita = yita.flatten() # yita 二维数组 -> 一维
        
        def poly2(x, a, b, c):
            return a*x**2 + b*x + c
        def poly3(x, a, b, c, d):
            return a*x**3 + b*x**2 + c*x + d
        def selfFunc(x, a, b):
            return a*x+b  # 2023.2.9: old, a*(x-c)**2 + b*(x-c)**3
        
        # get T0 from Quadratic polynomial
        optPoly2, covPoly2 = curve_fit(poly2, yita, tYita) # , p0=[10,1e3,1e6]
        optPoly3, covPoly3 = curve_fit(poly3, yita, tYita, p0=[1e3,1e3,1e3,1e5])
        optSfunc, covSfunc = curve_fit(selfFunc, yita, tYita) # 2023.2.9: use poly1

        # 2023.2.9: interpolate the y value of the onset time t0
        epoch_stamp = [t.timestamp() for t in df.index]
        f = interpolate.PchipInterpolator(epoch_stamp, df,extrapolate=True)

        fig, ax = plt.subplots(2,2, figsize=(10,8))
        ax[0,0].plot(yita, tYita, 'o',label = 'data')
        ax[0,0].plot(np.append(yita,0), poly2(np.append(yita,0), *optPoly2), label = 'fit: poly2, unc: %.2f s'% np.sqrt(np.diag(covPoly2))[2])
        ax[0,0].plot(np.append(yita,0), poly3(np.append(yita,0), *optPoly3), label = 'fit: poly3, unc: %.2f s'% np.sqrt(np.diag(covPoly3))[3])
        ax[0,0].plot(np.append(yita,0), selfFunc(np.append(yita,0), *optSfunc), label = 'fit: poly1, unc: %.2f s'% np.sqrt(np.diag(covSfunc))[1])
        # 2023.2.9: old self-defined function
        # ax[0,0].plot(selfFunc(tYita, *optSfunc), tYita, label = 'fit: self-defined, unc: %.2f s'% np.sqrt(np.diag(covSfunc))[2])
        ax[0,0].legend()

        # poly2
        # 2023.2.9: new fitting result to obtain t0
        t0_ploy2 = time[0]-timedelta(seconds=optPoly2[2])
        t0_flux_poly2 = f(t0_ploy2.timestamp())
        ax[0,1].plot(df.index, df.values, label = E)
        ax[0,1].plot(time, [i[1] for i in timeSeries], 'o')
        ax[0,1].plot(t0_ploy2, t0_flux_poly2, 'o', label='poly2')
        # 2023.2.9: old fitting result
        # ax[0,1].plot(datetime.fromtimestamp(optPoly2[2], tz = timezone.utc),df.values[np.abs(np.array([i.timestamp() for i in df.index])-optPoly2[2]).argmin()],
        #              'o', label='poly2') 
        ax[0,1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax[0,1].legend()

        # poly3
        t0_ploy3 = time[0]-timedelta(seconds=optPoly3[3])
        t0_flux_poly3 = f(t0_ploy3.timestamp())
        ax[1,0].plot(df.index, df.values, label = E)
        ax[1,0].plot(time, [i[1] for i in timeSeries], 'o')
        ax[1,0].plot(t0_ploy3, t0_flux_poly3, 'o', label='poly3')
        # ax[1,0].plot(datetime.fromtimestamp(optPoly3[3], tz = timezone.utc),df.values[np.abs(np.array([i.timestamp() for i in df.index])-optPoly3[3]).argmin()],
        #              'o', label='poly3')
        ax[1,0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax[1,0].legend()

        # self-defined
        t0_selfF = time[0]-timedelta(seconds=optSfunc[1])
        t0_flux_selfF = f(t0_selfF.timestamp())
        ax[1,1].plot(df.index, df.values, label = E)
        ax[1,1].plot(time, [i[1] for i in timeSeries], 'o')
        ax[1,1].plot(t0_selfF, t0_flux_selfF,'o', label='poly1')

        ax[1,1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax[1,1].legend()
        
        for i in np.arange(2):
            for j in np.arange(2):
                ax[i,j].autoscale(enable=True, axis='x', tight=True)
        
        if self.savefig == True:
            fig.savefig(self.figpath + '%s_T0 %.3s keV.png'%(self.NAME, E), dpi=300)
        if self.plotfig == True:
            plt.show()  

        # 2023.2.9: Save the coordinate of the onset time
        T0.loc['poly2',E] = [t0_ploy2, t0_flux_poly2]
        T0.loc['poly3',E] = [t0_ploy3, t0_flux_poly3]
        T0.loc['poly1',E] = [t0_selfF, t0_flux_selfF]

        T0_uncertainty.loc['poly2',E] = [np.sqrt(np.diag(covPoly2))[2]]
        T0_uncertainty.loc['poly3',E] = [np.sqrt(np.diag(covPoly3))[3]]
        T0_uncertainty.loc['poly1',E] = [np.sqrt(np.diag(covSfunc))[1]]
       
       
        return T0, T0_uncertainty
    
    def get_onsetTime(self):
        tYita = self.t_eta.copy(deep = True)
        temp = pd.DataFrame(columns = tYita.columns)
        tempUnc = pd.DataFrame(columns = tYita.columns)
        for i in range(len(self.polyorder_toFitOnsetTime)):
            temp.loc[0, temp.columns[i]] = self.T0.loc[self.polyorder_toFitOnsetTime[i], self.T0.columns[i]]
            tempUnc.loc[0, temp.columns[i]] = self.T0_uncertaintyFromDiffPoly.loc[self.polyorder_toFitOnsetTime[i], self.T0_uncertaintyFromDiffPoly.columns[i]]
        
        tYita.loc['0'] = temp.loc[0]
        self.onsetTime = tYita
        self.T0_uncertainty = tempUnc

    def calculate_releaseTime(self, eta='0', L=[]):
        """ Calculate electron release times based on path lengths. """
        pl = self.pathLength_tr
        t0 = np.array([i[0] for i in self.onsetTime.loc[eta]])
        # t0_unc = [timedelta(seconds=i[0]) for i in self.T0_uncertainty.loc[0]]
        

        if L:
            l1, l2, l3 = L
        else:
            l1 = max(pl.loc['0', 'pl'][0], 1.0)
            l2 = self.parker
            l3 = 0.5 * (l1 + l2) if abs(l2 - l1) > 0.5 else 2 * l2 - l1

        dt1 = np.array([timedelta(seconds=i) for i in l1 * SEEevent.au / (self.beta * 3 * 10**5)])
        dt2 = np.array([timedelta(seconds=i) for i in l2 * SEEevent.au / (self.beta * 3 * 10**5)])
        dt3 = np.array([timedelta(seconds=i) for i in l3 * SEEevent.au / (self.beta * 3 * 10**5)])

        self.Tr_approach2 = pd.DataFrame({'tr1':t0 - dt1, 'tr2':t0 - dt2, 'tr3':t0 - dt3 }, index = self.onsetTime.columns, columns=['tr1','tr2','tr3'])
        self.pathLength_forApproach2 = [l1, l2, l3]
        # return t0 - dt1, t0 - dt2, t0 - dt3

    def plot_releaseTimes(self, eta='0'):
        """ Plot the release times along with annotations and error bars. """

        energy = np.array([float(i.split('keV')[0]) for i in list(self.onsetTime.columns)])

        tr1 = self.Tr_approach2['tr1']
        tr2 = self.Tr_approach2['tr2']
        tr3 = self.Tr_approach2['tr3']
        l1,l2,l3 = self.pathLength_forApproach2
        t0_unc = [timedelta(seconds=i[0]) for i in self.T0_uncertainty.loc[0]]

        fig, ax = plt.subplots(1, figsize=(7, 6))
        fig.subplots_adjust(left=0.12, right=0.96, top=0.97, bottom=0.1)

        ax.errorbar(tr1, energy, xerr=t0_unc, marker='o', ls='none', label=f'{l1:.2f} AU')
        ax.errorbar(tr2, energy, xerr=t0_unc, marker='o', ls='none', label=f'{l2:.2f} AU (Parker)')
        ax.errorbar(tr3, energy, xerr=t0_unc, marker='o', ls='none', label=f'{l3:.2f} AU')

        # plot_additional_markers(ax)
        
        for m,n in zip(tr1, energy):
            ax.text(m+timedelta(seconds=0),n+energy.max()*1/100,m.strftime("%M:%S"), fontsize = 11,color='C0')
        for m,n in zip(tr2, energy):
            ax.text(m+timedelta(seconds=3),n-energy.max()*3/100,m.strftime("%M:%S"), fontsize = 11,color='C1')
        for m,n in zip(tr3, energy):
            ax.text(m,n+energy.max()/100,m.strftime("%M:%S"), fontsize = 11,color='C2')
        
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax.xaxis.grid(True, which='major', linestyle='--')
        ax.ticklabel_format(axis='y',scilimits=(2,2))
        ax.legend(loc='best')
        ax.set_xlabel('Time (UTC) $\eta$ = %s'%eta, fontsize = 17)
        ax.set_ylabel('Energy (keV)', fontsize=17)
        ax.tick_params(axis='both', labelsize=13)
        
        # if self.savefig:
        #     fig.savefig(self.figpath + '%s_eta%s_Tr.jpg'%(self.NAME,eta), dpi=300)
        
        # if self.plotfig:
        #     plt.show()

        return fig, ax

    
    def export_toExcel(self, resultFilePath = None):
        """
        Exports various attributes of self.event to an Excel file with multiple sheets.

        Parameters:
        
        """
        if resultFilePath is None:
            resultFilePath = self.figpath
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(resultFilePath + self.NAME + '.xlsx', engine='xlsxwriter')
        
        # Write DataFrames to different sheets in the Excel file
        self.pathLength_tr_approach1.to_excel(writer, sheet_name='FVDA_approach1')
        self.t_eta.to_excel(writer, sheet_name='t(eta)')
        self.onsetTime.iloc[-1].to_excel(writer, sheet_name='(onsetTime, intensity)')
        self.T0_uncertainty.to_excel(writer, sheet_name='onsetTime_uncertainty')

        self.pathLength_tr.to_excel(writer, sheet_name='PathLength_Tr_atEachEta(VDA)')


        # Convert datetime to naive before writing to Excel
        for i in self.Tr_approach2:
            self.Tr_approach2[i] = self.Tr_approach2[i].dt.tz_localize(None)
        self.Tr_approach2.to_excel(writer, sheet_name='Tr_FVDA_approach2')
        pl = pd.DataFrame(np.round(self.pathLength_forApproach2,3).reshape(1,3),columns=['PL1','PL2','PL3'])
        pl.to_excel(writer, sheet_name='PathLength_usedInFvdaApproach2')

        # Close the Pandas Excel writer and output the Excel file.
        writer.close()
