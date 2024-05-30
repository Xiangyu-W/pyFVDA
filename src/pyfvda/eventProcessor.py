import os
import pandas as pd
import numpy as np
from SEP_analysis import SEEevent
from eventTimingGroup import EventTimingGroup
import matplotlib.pyplot as plt
import pickle

def load_event_list(file_path):
    """Load event data from an Excel file."""
    return pd.read_excel(file_path)

class EventProcessor:
    def __init__(self, event_list, idx):
        """Initialize the event pro with an event list and an index."""
        self.event_list = event_list
        self.idx = idx
        self.eventPath = ''
        print(f"Event_{self.event_list.loc[ self.idx-1,'SC']}({self.event_list.loc[ self.idx-1,'start time']}) is selected.")
    
    def load_fromPickle(file_path):
        """Deserializes and loads an EventProcessor instance from a pickle file.
        Parameters:
        file_path : str
            The file path from which to load the pickle file.
            
        Returns:
        EventProcessor
            The EventProcessor instance loaded from the pickle file.
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def createEventObject(self, plotfig=True, savefig=False,figpath=None):
        """Create and configure the CDF object for the event."""
        #  Event name & path
        event_info = self.event_list.loc[self.idx - 1]
        t_range = [i.strftime('%Y-%m-%dT%H:%M:%SZ') for i in event_info[['start time', 'end time']].to_list()]
        directionID = [event_info['pitch angle bin']]
        SC = event_info['SC']
        onsetHour = event_info['hour']
        
        event_name = f"{SC}_{event_info['start time'].strftime('%Y-%m-%d %H_%M')}_{str(event_info['class'])}"
        
        if figpath is None:
            figpath = './Event/%s/'%event_name
        os.makedirs(figpath, exist_ok=True)
        self.eventPath = figpath

        # generate CDF class object.
        self.event = SEEevent(event_name, directionID, t_range, SC)
        self.event.figpath = figpath
        self.event.plotfig = plotfig
        self.event.savefig = savefig
        self.event.onsetHour = onsetHour
        return self.event

    def getSolarWindSpeed(self, t_vsw = []):
        """Perform the solar wind analysis on the event data."""
        # get vsw and calc parker spiral
        self.event.solarWind(t_vsw)

    def prepare_data(self, initial_WindowSize = [7]*6, polyorder = [2]*6, timeForFvda = []):
        """Smooth the data for analysis."""

        # This will store the smoothed data in self.dataF.
        # event.Filter(initial_WindowSize, polyorder, timeForFvda = [str, str])
        # plot_button = self.event.Filter([19]*6, polyorder=[2]*6, timeForFvda = ['2021-10-09 06:00:00', '2021-10-09 10:00:00'])
        plot_button = self.event.Filter(initial_WindowSize, polyorder, timeForFvda)
        print('!Remember to Specify: "bins, background range, peak range" before call the "getTime_eta" method.')
        return plot_button # Return the plot button is to ensure the interactive interface can be used.

    def getTime_eta(self, bgRange, peakRange, binsForFvda, eta, specialBg={}):
        """Get the time (eta) for the event."""
        '''
        bgRange=['2021-10-09 06:10', '2021-10-09 06:20'] 
        peakRange=['2021-10-09 07:30', '2021-10-09 08:20'] 
        binsForFvda=[1, 2, 3, 4, 5, 6]
        eta= np.array([np.arange(0.7, 0.1, -0.05)]).T
        specialBg={'179.0 keV':['2013-05-13 16:10:00','2013-05-13 16:20:00'],'49.7 keV':['2013-05-13 16:20:00','2013-05-13 16:30:00']}
        '''
        self.event.setParameters_for_getTime_eta(bgRange, peakRange, binsForFvda, eta, specialBg)
        self.event.getT_Eta()
    
    def getOnsetTime(self, polyorder_toFitOnsetTime = [1]*6):
        """Get the onset time for the event."""
        self.event.polyorder_toFitOnsetTime = ['poly%d'%i for i in polyorder_toFitOnsetTime]
        self.event.get_onsetTime()
    
    def display_figpath(self):
        """Display the figure path where results are stored."""
        print(self.eventPath)

    def fvda_approach1(self, eta_forDisplay = [0.7, 0.4, 0.15, 0], etaRangeForFitting = np.arange(0.7,0.1,-0.05)):
        """FVDA approach 1."""
        '''
        eta_forDisplay: eta for display in the deltaTime - 1/beta fitting plot. This fitting is to get Pathlength(eta!=0), Tr(eta!=0).
        etaRangeForFitting: eta for Pathlength-eta/ Tr-eta fitting. Aim: to get Pathlength(eta=0), Tr(eta=0).
            So, etaRangeForFitting should not include eta=0. Otherwise, the result will be wrong. 
        '''
        eta_forDisplay_str = [f'{i:.2f}' if i!=0 else '0' for i in eta_forDisplay]
        self.event.plotFittingResult_pathLength_eta(eta = eta_forDisplay)
        self.event.plot_FVDA_approach1(etaForDisplay_pathLength = eta_forDisplay_str, etaRangeForFitting = etaRangeForFitting)


    def fvda_approach2(self, eta='0', pathLength_forApproach2=[], plotExtraData = None):
        """FVDA approach 2."""
        # plot Tr from approach 2
        self.event.calculate_releaseTime(eta, pathLength_forApproach2)
        fig,ax = self.event.plot_releaseTimes(eta)
        
        # plot extra data in the same figure
        if plotExtraData is not None:
            for obj in plotExtraData.values():
                obj.plot_on_axes(ax)
            # plotExtraData.plot_on_axes(ax)
        
        # Save the figure
        if self.event.savefig:
            fig.savefig(self.event.figpath + '%s_eta%s_Tr.jpg'%(self.event.NAME, eta), dpi=300)
        if self.event.plotfig:
            plt.show()
    

    def exportResultToExcel(self, resultFilePath = None):
        """
        Exports various attributes of self.event to an Excel file with multiple sheets.

        Parameters:
        self.event: Object containing event data.
        """
        self.event.export_toExcel(resultFilePath)

    def saveResultToPickle(self, resultFilePath = None):
        """Serializes and saves the current instance to a pickle file.
        Parameters:
        file_path : str
            The file path where the pickle file will be saved.
        """
        if resultFilePath is None:
            resultFilePath = self.event.figpath 
        
        with open(resultFilePath + self.event.NAME + '_EventProcessor.pkl', 'wb') as f:
            pickle.dump(self, f)


# Example usage of the code
if __name__ == "__main__":
    event_list = load_event_list('./eventListForTest.xlsx')
    pro = EventProcessor(event_list, idx=5)
    # pro.display_figpath()
    
    event = pro.createEventObject(plotfig=True, savefig=False)
    pro.getSolarWindSpeed() # Stored in pro.event.vsw & pro.event.parker

    pro.prepare_data()
    pro.getTime_eta(bgRange=['2021-10-09 06:10', '2021-10-09 06:20'], 
                    peakRange=['2021-10-09 07:30', '2021-10-09 08:20'], 
                    binsForFvda=[1, 2, 3, 4, 5, 6], 
                    eta= np.array([np.arange(0.7, 0.1, -0.05)]).T)
    pro.getOnsetTime()

    # FVDA approach 1
    # will get event.pathLength_tr
    # pro.fvda_approach1(eta_forDisplay = [0.7, 0.4, 0.3, 0.15])
    pro.fvda_approach1(etaRangeForFitting = np.arange(0.7,0.1,-0.05))
    

    # Create a group of timing from other observations, e.g., HXR, type III bursts
    hxrTime = [ # (time, y_value, label)
        ("2021-10-09 06:45:00", 25, "HXR 25keV"),
        ("2021-10-09 06:47:00", 50, "HXR 50keV")
        ] 
    hxrGroup = EventTimingGroup(hxrTime, source="RHESSI", isReleaseTime = True, radial_distance=1.0, color='r', marker='d')
    pro.extraData = {'HXR': hxrGroup}

    # FVDA approach 2
    # will get event.Tr_approach2, event.pathLength_forApproach2
    pro.fvda_approach2(plotExtraData=hxrGroup)

    # # Export to Excel
    # pro.exportResultToExcel()
    # pro.saveResultToPickle()

    # # Load the EventProcessor instance from the pickle file
    # processorObj = EventProcessor.load_fromPickle(file_path = './Event/STA_2021-10-09 06_00_nan/STA_2021-10-09 06_00_nanEventProcessor.pkl')
