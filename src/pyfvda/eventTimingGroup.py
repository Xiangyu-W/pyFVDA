from datetime import timedelta
import pandas as pd

class EventTimingGroup:
    au = 1.4959787e8 # km

    def __init__(self, eventTiming, source, isReleaseTime = False, radial_distance=0., travelSpeed = 3*10**5, errorbar_sec=30, color='r', marker='*'):
        """
        Initialize an EventTimingGroup object with necessary parameters.

        :param eventTiming: Datetime or list of datetime objects indicating the event time.
        :param source: String indicating the source of the event data (e.g., 'RHESSI', 'FERMI').
        :param radial_distance: The radial distance from the Sun in astronomical units (AU). 
        :param travelSpeed: Speed of the particle/event to calculate travel time (in km/s).
        :param errorbar_sec: Seconds for error bar width.
        :param isReleaseTime: Boolean indicating if the event time is the release time.
        :param color: (Optional) Marker color in the plot.
        :param marker: (Optional) Marker style in the plot.
        """
        self.times = [(pd.to_datetime(time), y_value, label, isReleaseTime) for time, y_value, label in eventTiming]
        self.sc = source
        self.radial_distance = radial_distance
        self.travelSpeed = travelSpeed
        self.errorbar_sec = errorbar_sec

        self.color = color
        self.marker = marker


    def add_event(self, time, y_value, label, isReleaseTime=False):
        """ Add an event to the group. """
        self.times.append((pd.to_datetime(time), y_value, label, isReleaseTime))

    def delete_event(self, index):
        """
        Delete an event from the group by index.

        :param index: The index of the event in the list to be removed.
        """
        if 0 <= index < len(self.times):
            del self.times[index]
        else:
            raise IndexError("Event index out of range.")

    def calculate_ReleaseTime(self, time):
        """Calculate the release time of the event based on the travel time.
        travelSpeed: km/s
        """
        travel_time = timedelta(seconds=(self.radial_distance * EventTimingGroup.au) / self.travelSpeed)
        return time - travel_time


    def plot_on_axes(self, ax):
        """
        Plot all events in the group on given matplotlib axes, adjusting for travel time if necessary.
        """
        for time, y_value, label, isReleaseTime in self.times:
            releaseTime = self.calculate_ReleaseTime(time) if not isReleaseTime else time

            # Plot each event
            ax.errorbar(releaseTime, y_value, xerr=timedelta(seconds=self.errorbar_sec), 
                        marker=self.marker, ls='none', label=f'{label}', color=self.color)
            ax.text(releaseTime + timedelta(seconds=2), y_value+5,
                    f'{releaseTime.strftime("%H:%M:%S")}', color=self.color, fontsize=11)
            
            ax.legend()
