import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.frequencies import to_offset
from typing import Tuple, Optional, Callable, Any, List


class MultivariateTimeSeries:
    """
        A TimeSeries is an immutable object defined by the following components.

        unique multivariate time series, with confidence interval and discrete features

        :param series: The actual time series, as a pandas Series with a proper time index.
        :param confidence_lo: Optionally, a Pandas Series representing lower confidence interval.
        :param confidence_hi: Optionally, a Pandas Series representing upper confidence interval.

        Within this class, TimeSeries type annotations are 'TimeSeries'; see:
        https://stackoverflow.com/questions/15853469/putting-current-class-as-return-type-annotation
    """

    def __init__(self, series: pd.DataFrame, date_index: str = None, feats: [str, List[str]] = None, confidence_lo: [str, List[str]] = None,
                 confidence_hi: [str, List[str]] = None, holiday: [str, pd.Series] = None, discrete: list = [], labels: np.ndarray = None):

        if type(confidence_lo) is not list:
            confidence_lo = [confidence_lo]
        if type(confidence_hi) is not list:
            confidence_hi = [confidence_hi]
        assert len(series) >= 3, 'Series must have at least three values.'
        if date_index is None:
            assert isinstance(series.index, pd.DatetimeIndex), 'Series must be indexed with a DatetimeIndex.'
        else:
            series = series.set_index(pd.DatetimeIndex(series[date_index]))
            series = series.drop(date_index, axis=1)
        for series_dtype in series.dtypes.values:
            assert np.issubdtype(series_dtype, np.number), 'Series must contain numerical values.'

        self._dataframe: pd.DataFrame = series.sort_index()  # Sort by time
        self._freq: str = self._dataframe.index.inferred_freq  # Infer frequency

        # TODO: optionally fill holes (including missing dates) - for now we assume no missing dates
        assert self._freq is not None, 'Could not infer frequency. Are some dates missing? Is Series too short (n=2)?'

        # TODO: are there some pandas Series where the line below causes issues?
        self._dataframe.index.freq = self._freq  # Set the inferred frequency in the Pandas series

        # check number of features in time series
        if feats is None:
            feats = self._dataframe.columns.drop([*confidence_lo, *confidence_hi, holiday, *discrete], errors='ignore')
        self._nfeat = 1 if type(feats) is str else len(feats)
        col_mapper = {feat: 'f{}'.format(i) for i, feat in enumerate(feats)}
        self._dataframe.rename(columns=col_mapper, inplace=True)

        # Handle confidence intervals:
        conf_mapper = {feat: 'cl{}'.format(i) for i, feat in enumerate(confidence_lo)}
        conf_mapper.update({feat: 'ch{}'.format(i) for i, feat in enumerate(confidence_hi)})
        self._dataframe.rename(columns=conf_mapper, inplace=True)
        self._confidence_lo = self._dataframe.columns[self._dataframe.columns.str.startswith('cl')].values.tolist()
        self._confidence_hi = self._dataframe.columns[self._dataframe.columns.str.startswith('ch')].values.tolist()
        # Handle discrete features (labels)
        self._is_holiday = holiday
        self._discrete_features = discrete
        self._features = ['f{}'.format(i) for i in range(self._nfeat)]

        # reorder logically the dataframe (fx, clx, chx, dy, h)
        col_names = self._features + self._confidence_lo + self._confidence_hi + discrete
        col_names += [] if holiday is None else [holiday]
        self._dataframe = self._dataframe[col_names]

    def pd_dataframe(self, s: [slice, int] = None) -> pd.DataFrame:
        """
        Returns the underlying Pandas Dataframe of these TimeSeries.

        :return: A Pandas Dataframe.
        """
        if s is None:
            s = slice(None, self._nfeat)
        elif type(s) is int:
            assert self._nfeat > s >= 0, "The desired time series must be comprised between 0 and `nfeat`"
        elif type(s) is slice:
            assert self._nfeat > s.stop and s.start >= 0, "The desired time series must be comprised" \
                                                          " between 0 and `nfeat`"
        return self._dataframe.iloc[:, s]

    def conf_lo_pd_series(self) -> Optional[pd.Series]:
        """
        Returns the underlying Pandas Dataframe of the lower confidence intervals if they exists.

        :return: A Pandas Dataframe for the lower confidence intervals.
        """
        return self._dataframe[self._confidence_lo] if self._confidence_lo else None

    def conf_hi_pd_series(self) -> Optional[pd.Series]:
        """
        Returns the underlying Pandas Series of the upper confidence interval if it exists.

        :return: A Pandas Series for the upper confidence interval.
        """
        return self._dataframe[self._confidence_hi] if self._confidence_hi else None

    def start_time(self) -> pd.Timestamp:
        """
        Returns the start time of the time index.

        :return: A timestamp containing the first time of the TimeSeries.
        """
        return self._dataframe.index[0]

    def end_time(self) -> pd.Timestamp:
        """
        Returns the end time of the time index.

        :return: A timestamp containing the last time of the TimeSeries.
        """
        return self._dataframe.index[-1]

    def values(self) -> np.ndarray:
        """
        Returns the values of the TimeSeries.

        :return: A numpy array containing the values of the TimeSeries.
        """
        return self._dataframe.values[:, :self._nfeat]

    def time_index(self) -> pd.DatetimeIndex:
        """
        Returns the index of the TimeSeries.

        :return: A DatetimeIndex containing the index of the TimeSeries.
        """
        return self._dataframe.index

    def freq(self) -> pd.DateOffset:
        """
        Returns the frequency of the TimeSeries.

        :return: A DateOffset with the frequency.
        """
        return to_offset(self._freq)

    def freq_str(self) -> str:
        """
        Returns the frequency of the TimeSeries.

        :return: A string with the frequency.
        """
        return self._freq

    def duration(self) -> pd.Timedelta:
        """
        Returns the duration of the TimeSeries.

        :return: A Timedelta of the duration of the TimeSeries.
        """
        return self._dataframe.index[-1] - self._dataframe.index[0]

    def copy(self):
        """
        Make a copy of this object time series
        :param deep: Make a deep copy. If False, the Series will be the same
        :return: A copy of the TimeSeries
        """
        return MultivariateTimeSeries(self._dataframe, feats=self._features,
                                      confidence_lo=self._confidence_lo, confidence_hi=self._confidence_hi,
                                      holiday=self._is_holiday, discrete=self._discrete_features)

    def _raise_if_not_within(self, ts: pd.Timestamp):

        if (ts < self.start_time()) or (ts > self.end_time()):
            raise ValueError('Timestamp must be between {} and {}'.format(self.start_time(), self.end_time()))

    def split_after(self, ts: pd.Timestamp) -> Tuple['TimeSeries', 'TimeSeries']:
        """
        Splits the TimeSeries in two, around a provided timestamp [ts].

        The timestamp may not be in the TimeSeries. If it is, the timestamp will be included in the
        first of the two TimeSeries, and not in the second.

        :param ts: The timestamp that indicates the splitting time.
        :return: A tuple (s1, s2) of TimeSeries with indices smaller or equal to [ts]
                 and greater than [ts] respectively.
        """

        self._raise_if_not_within(ts)

        ts = self.time_index()[self.time_index() <= ts][-1]  # closest index before ts (new ts)

        start_second_series: pd.Timestamp = ts + self.freq()  # second series does not include ts
        return self.slice(self.start_time(), ts), self.slice(start_second_series, self.end_time())

    def split_before(self, ts: pd.Timestamp) -> Tuple['TimeSeries', 'TimeSeries']:
        """
        Splits a TimeSeries in two, around a provided timestamp [ts].

        The timestamp may not be in the TimeSeries. If it is, the timestamp will be included in the
        second of the two TimeSeries, and not in the first.

        :param ts: The timestamp that indicates the splitting time.
        :return: A tuple (s1, s2) of TimeSeries with indices smaller than [ts]
                 and greater or equal to [ts] respectively.
        """
        self._raise_if_not_within(ts)

        ts = self.time_index()[self.time_index() >= ts][0]  # closest index after ts (new ts)

        end_first_series: pd.Timestamp = ts - self.freq()  # second series does not include ts
        return self.slice(self.start_time(), end_first_series), self.slice(ts, self.end_time())

    def drop_after(self, ts: pd.Timestamp) -> 'TimeSeries':
        """
        Drops everything after the provided timestamp [ts], included.

        The timestamp may not be in the TimeSeries. If it is, the timestamp will be dropped.

        :param ts: The timestamp that indicates cut-off time.
        :return: A new TimeSeries, with indices smaller than [ts].
        """
        self._raise_if_not_within(ts)

        ts = self.time_index()[self.time_index() >= ts][0]  # closest index after ts (new ts)

        end_series: pd.Timestamp = ts - self.freq()  # new series does not include ts
        return self.slice(self.start_time(), end_series)

    def drop_before(self, ts: pd.Timestamp) -> 'TimeSeries':
        """
        Drops everything before the provided timestamp [ts], included.

        The timestamp may not be in the TimeSeries. If it is, the timestamp will be dropped.

        :param ts: The timestamp that indicates cut-off time.
        :return: A new TimeSeries, with indices greater than [ts].
        """
        self._raise_if_not_within(ts)

        ts = self.time_index()[self.time_index() <= ts][-1]  # closest index before ts (new ts)

        start_series: pd.Timestamp = ts + self.freq()  # new series does not include ts
        return self.slice(start_series, self.end_time())

    def slice(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> 'TimeSeries':
        """
        Returns a new TimeSeries, starting later than [start_ts] and ending before [end_ts], inclusive on both ends.

        The timestamps may not be in the time series. TODO: should new timestamps be added? Think not

        :param start_ts: The timestamp that indicates the left cut-off.
        :param end_ts: The timestamp that indicates the right cut-off.
        :return: A new TimeSeries, which indices greater or equal than [start_ts] and smaller or equal than [end_ts].
        """

        assert end_ts > start_ts, 'End timestamp must be strictly after start timestamp when slicing.'
        assert end_ts >= self.start_time(), 'End timestamp must be after the start of the time series when slicing.'
        assert start_ts <= self.end_time(), 'Start timestamp must be after the end of the time series when slicing.'

        dataframe_sliced = self._dataframe[self._dataframe.index >= start_ts]
        dataframe_sliced = dataframe_sliced[dataframe_sliced.index <= end_ts]
        return MultivariateTimeSeries(dataframe_sliced, feats=self._features,
                                      confidence_lo=self._confidence_lo, confidence_hi=self._confidence_hi,
                                      holiday=self._is_holiday, discrete=self._discrete_features)

    def slice_n_points_after(self, start_ts: pd.Timestamp, n: int) -> 'TimeSeries':
        """
        Returns a new TimeSeries, starting later than [start_ts] (included) and having (at most) [n] points.

        The timestamp may not be in the time series. If it is, it will be included in the new TimeSeries.

        :param start_ts: The timestamp that indicates the splitting time.
        :param n: The maximal length of the new TimeSeries.
        :return: A new TimeSeries, with length at most [n] and indices greater or equal than [start_ts].
        """

        assert n >= 0, 'n should be a positive integer.'  # TODO: logically raise if n<3, cf. init

        self._raise_if_not_within(start_ts)

        start_ts = self.time_index()[self.time_index() >= start_ts][0]  # closest index after start_ts (new start_ts)

        end_ts: pd.Timestamp = start_ts + (n - 1) * self.freq()  # (n-1) because slice() is inclusive on both sides
        return self.slice(start_ts, end_ts)

    def slice_n_points_before(self, end_ts: pd.Timestamp, n: int) -> 'TimeSeries':
        """
        Returns a new TimeSeries, ending before [end_ts] (included) and having (at most) [n] points.

        The timestamp may not be in the TimeSeries. If it is, it will be included in the new TimeSeries.

        :param end_ts: The timestamp that indicates the splitting time.
        :param n: The maximal length of the new time series.
        :return: A new TimeSeries, with length at most [n] and indices smaller or equal than [end_ts].
        """

        assert n >= 0, 'n should be a positive integer.'

        self._raise_if_not_within(end_ts)

        end_ts = self.time_index()[self.time_index() <= end_ts][-1]

        start_ts: pd.Timestamp = end_ts - (n - 1) * self.freq()  # (n-1) because slice() is inclusive on both sides
        return self.slice(start_ts, end_ts)

    def intersect(self, other: 'TimeSeries') -> 'TimeSeries':
        """
        Returns a slice containing the intersection of this TimeSeries and the one provided in argument.

        This slice can be used with the `__getitem__` method.

        :param other: A second TimeSeries.
        :return: A pd.DateTimeIndex containing the dates in the intersection of the two TimeSeries.
        """

        return self.time_index().intersection(other.time_index())

    # TODO: other rescale? such as giving a ratio, or a specific position? Can be the same function
    def rescale_with_value(self, value_at_first_step: float) -> 'TimeSeries':
        """
        Returns a new TimeSeries, which is a multiple of this TimeSeries such that
        the first value is [value_at_first_step].
        Numerical imprecisions appear with [value_at_first_step] > 1e+24
        TODO: can receive array of first values

        :param value_at_first_step: The new value for the first entry of the TimeSeries.
        :return: A new TimeSeries, whose first value was changed to [value_at_first_step] and whose others values
                have been scaled accordingly.
        """

        assert self.values()[0, 0] != 0, 'Cannot rescale with first value 0.'

        coef = value_at_first_step / self.values()[0, 0]  # TODO: should the new TimeSeries have the same dtype?
        new_series = self._dataframe.copy()
        new_series[self._features] = coef * new_series[self._features]
        new_series[self._confidence_lo] = coef * new_series[self._confidence_lo]
        new_series[self._confidence_hi] = coef * new_series[self._confidence_hi]
        return MultivariateTimeSeries(new_series, feats=self._features,
                                      confidence_hi=self._confidence_hi, confidence_lo=self._confidence_lo,
                                      holiday=self._is_holiday, discrete=self._discrete_features)

    def shift(self, n: int) -> 'TimeSeries':
        """
        Shifts the time axis of this TimeSeries by [n] time steps.

        If n > 0, shifts in the future. If n < 0, shifts in the past.

        For example, with n=2 and freq='M', March 2013 becomes May 2013. With n=-2, March 2013 becomes Jan 2013.

        :param n: The signed number of time steps to shift by.
        :return: A new TimeSeries, with a shifted index.
        """
        # TODO: no error raised if freq is different than day and overflow happens, so raise it ourselves
        try:
            self.time_index()[-1] + n * self.freq()
        except pd.errors.OutOfBoundsDatetime:
            raise OverflowError("the add operation between {} and {} will overflow".format(n * self.freq(),
                                                                                           self.time_index()[-1]))
        new_time_index = self._dataframe.index.map(lambda ts: ts + n * self.freq())
        new_series = self._dataframe.copy()
        new_series.index = new_time_index
        return MultivariateTimeSeries(new_series, feats=self._features,
                                      confidence_hi=self._confidence_hi, confidence_lo=self._confidence_lo,
                                      holiday=self._is_holiday, discrete=self._discrete_features)

    @staticmethod
    def from_times_and_values(times: pd.DatetimeIndex,
                              values: np.ndarray,
                              confidence_lo: np.ndarray = None,
                              confidence_hi: np.ndarray = None,
                              holiday: np.ndarray = None,
                              discrete: np.ndarray = None) -> 'MultivariateTimeSeries':
        """
        Returns a TimeSeries built from an index and values.

        :param times: A DateTimeIndex for the TimeSeries.
        :param values: An array of values for the TimeSeries.
        :param confidence_lo: The lower confidence interval values (optional).
        :param confidence_hi: The higher confidence interval values (optional).
        :return: A TimeSeries constructed from the inputs.
        """

        series = pd.DataFrame(values, index=times)
        features = series.columns
        conf_lo = ['a{}'.format(i) for i in range(confidence_lo.shape[1])] if confidence_lo is not None else None
        conf_hi = ['b{}'.format(i) for i in range(confidence_hi.shape[1])] if confidence_hi is not None else None
        is_holiday = 'holiday' if holiday is not None else None
        discrete_features = ['d{}'.format(i) for i in range(discrete.shape[1])] if discrete is not None else []
        df_lo = pd.DataFrame(confidence_lo, index=times, columns=conf_lo) if confidence_lo is not None else None
        df_hi = pd.DataFrame(confidence_hi, index=times, columns=conf_hi) if confidence_hi is not None else None
        df_holy = pd.DataFrame(holiday, index=times, columns=[is_holiday]) if holiday is not None else None
        df_discrete = pd.DataFrame(discrete, index=times, columns=discrete_features) if discrete is not None else None

        for df in [df_lo, df_hi, df_holy, df_discrete]:
            if df is None:
                continue
            series = series.T.append(df.T).T

        return MultivariateTimeSeries(series, feats=features,
                                      confidence_lo=conf_lo, confidence_hi=conf_hi,
                                      holiday=is_holiday, discrete=discrete_features)

    # TODO: from list of series?

    def plot(self, *args, plot_ci=True, **kwargs):
        """
        Currently this is just a wrapper around pd.Series.plot()
        """
        # temporary work-around for the pandas.plot issue
        # errors = self._combine_or_none(self._confidence_lo, self._confidence_hi,
        #                                lambda x, y: np.vstack([x.values, y.values]))
        # self._series.plot(yerr=errors, *args, **kwargs)
        plt.plot(self.time_index(), self.values(), *args, **kwargs)
        x_label = self.time_index().name
        if x_label is not None and len(x_label) > 0:
            plt.xlabel(x_label)
        # TODO: use pandas plot in the future
        if plot_ci and self._confidence_lo is not None and self._confidence_hi is not None:
            plt.fill_between(self.time_index(), self._confidence_lo.values, self._confidence_hi.values, alpha=0.5)

    """
    Some useful methods for TimeSeries combination:
    """

    def has_same_time_as(self, other: 'TimeSeries') -> bool:
        """
        Checks whether this TimeSeries and another one have the same index.

        :param other: A second TimeSeries.
        :return: A boolean. True if both TimeSeries have the same index, False otherwise.
        """

        if self.__len__() != len(other):
            return False
        return (other.time_index() == self.time_index()).all()

    # TODO: is union function useful too?
    # TODO: should append only at the end of the series? or can we create holes and "interpolate" their values?
    def append(self, other: 'TimeSeries') -> 'TimeSeries':
        """
        Appends another TimeSeries to this TimeSeries.

        :param other: A second TimeSeries.
        :return: A new TimeSeries, obtained by appending the second TimeSeries to the first.
        """

        assert other.start_time() == self.end_time() + self.freq(), 'Appended TimeSeries must start one time step ' \
                                                                    'after current one.'
        # TODO additional check?
        assert other.freq() == self.freq(), 'Appended TimeSeries must have the same frequency as the current one'

        series = self._series.append(other.pd_series())
        conf_lo = None
        conf_hi = None
        if self._confidence_lo is not None and other.conf_lo_pd_series() is not None:
            conf_lo = self._confidence_lo.append(other.conf_lo_pd_series())
        if self._confidence_hi is not None and other.conf_hi_pd_series() is not None:
            conf_hi = self._confidence_hi.append(other.conf_hi_pd_series())
        return TimeSeries(series, conf_lo, conf_hi)

    def append_values(self, values: np.ndarray, index: pd.DatetimeIndex = None,
                      conf_lo: np.ndarray = None, conf_hi: np.ndarray = None) -> 'TimeSeries':
        """
        Appends values to current TimeSeries, to the given indices.

        If no index is provided, assumes that it follows the original data.
        Does not add new confidence values if there were none first.
        Does not update value if already existing indices are provided.

        :param values: An array with the values to append.
        :param index: A DateTimeIndex for each value (optional).
        :param conf_lo: The lower confidence interval values (optional).
        :param conf_hi: The higher confidence interval values (optional).
        :return: A new TimeSeries with the new values appended
        """
        if len(values) < 1:
            return self
        if isinstance(values, list):
            values = np.array(values)
        if index is None:
            index = pd.DatetimeIndex([self.end_time() + i * self.freq() for i in range(1, 1 + len(values))])
        assert isinstance(index, pd.DatetimeIndex), 'values must be indexed with a DatetimeIndex.'
        assert len(index) == len(values)
        assert self.time_index().intersection(index).empty, "cannot add already present time index"
        new_indices = index.argsort()
        index = index[new_indices]
        # TODO do we really want that?
        assert index[0] == self.end_time() + self.freq(), 'Appended index must start one time step ' \
                                                          'after current one.'
        if len(index) > 2:
            assert index.inferred_freq == self.freq_str(), 'Appended index must have ' \
                                                           'the same frequency as the current one'
        elif len(index) == 2:
            assert index[-1] == index[0] + self.freq(), 'Appended index must have ' \
                                                        'the same frequency as the current one'
        values = values[new_indices]
        new_series = pd.Series(values, index=index)
        series = self._series.append(new_series)
        if conf_lo is not None and self._confidence_lo is not None:
            assert len(index) == len(conf_lo)
            conf_lo = conf_lo[new_indices]
            conf_lo = self._confidence_lo.append(pd.Series(conf_lo, index=index))
        if conf_hi is not None and self._confidence_hi is not None:
            assert len(index) == len(conf_hi)
            conf_hi = conf_hi[new_indices]
            conf_hi = self._confidence_hi.append(pd.Series(conf_hi, index=index))

        return TimeSeries(series, conf_lo, conf_hi)

    def update(self, index: pd.DatetimeIndex, values: np.ndarray = None,
               conf_lo: np.ndarray = None, conf_hi: np.ndarray = None, inplace: bool = True) -> 'TimeSeries':
        """
        Updates the Series with the new values provided.
        If indices are not in original TimeSeries, they will be discarded.
        At least one parameter other than index must be filled.
        Use np.nan to ignore a specific index in a series.

        It will raise an error if try to update a missing CI series

        :param index: A DateTimeIndex containing the indices to replace.
        :param values: An array containing the values to replace (optional).
        :param conf_lo: The lower confidence interval values to change (optional).
        :param conf_hi: The higher confidence interval values (optional).
        :param inplace: If True, do operation inplace and return None, defaults to True.
        :return: A TimeSeries with values updated
        """
        assert not (values is None and conf_lo is None and conf_hi is None), "At least one parameter must be filled " \
                                                                             "other than index"
        assert True if values is None else len(values) == len(index), \
            "The number of values must correspond to the number of indices: {} != {}".format(len(values), len(index))
        assert True if conf_lo is None else len(conf_lo) == len(index), \
            "The number of values must correspond to the number of indices: {} != {}".format(len(conf_lo), len(index))
        assert True if conf_hi is None else len(conf_hi) == len(index), \
            "The number of values must correspond to the number of indices: {} != {}".format(len(conf_hi), len(index))
        ignored_indices = [index.get_loc(ind) for ind in (set(index) - set(self.time_index()))]
        index = index.delete(ignored_indices)
        series = values if values is None else pd.Series(np.delete(values, ignored_indices), index=index)
        conf_lo = conf_lo if conf_lo is None else pd.Series(np.delete(conf_lo, ignored_indices), index=index)
        conf_hi = conf_hi if conf_hi is None else pd.Series(np.delete(conf_hi, ignored_indices), index=index)
        assert len(index) > 0, "must give at least one correct index"
        if inplace:
            if series is not None:
                self._series.update(series)
            if conf_lo is not None:
                self._confidence_lo.update(conf_lo)
            if conf_hi is not None:
                self._confidence_hi.update(conf_hi)
            return None
        else:
            new_series = self.pd_series()
            new_lo = self.conf_lo_pd_series()
            new_hi = self.conf_hi_pd_series()
            if series is not None:
                new_series.update(series)
            if conf_lo is not None:
                new_lo.update(conf_lo)
            if conf_hi is not None:
                new_hi.update(conf_hi)
            return TimeSeries(new_series, new_lo, new_hi)

    def drop_values(self, index: pd.DatetimeIndex, inplace: bool = True, **kwargs):
        """
        Remove elements of all series with specified indices.

        :param index: The indices to be dropped
        :param kwargs: Option to pass to pd.Series drop method
        :param inplace: If True, do operation inplace and return None, defaults to True.
        :return: A TimeSeries with values dropped
        """
        series = self._series.drop(index=index, inplace=inplace, **kwargs)
        conf_lo = self._op_or_none(self._confidence_lo, lambda s: s.drop(index, inplace=inplace, **kwargs))
        conf_hi = self._op_or_none(self._confidence_hi, lambda s: s.drop(index, inplace=inplace, **kwargs))
        if inplace:
            return None
        return TimeSeries(series, conf_lo, conf_hi)

    @staticmethod
    def _combine_or_none(series_a: Optional[pd.Series],
                         series_b: Optional[pd.Series],
                         combine_fn: Callable[[pd.Series, pd.Series], Any]):
        """
        Combines two Pandas Series [series_a] and [series_b] using [combine_fn] if neither is None.

        :param series_a: A Pandas Series.
        :param series_b: A Pandas Series.
        :param combine_fn: An operation with input two Pandas Series and output one Pandas Series.
        :return: A new Pandas Series, the result of [combine_fn], or None.
        """

        if series_a is not None and series_b is not None:
            return combine_fn(series_a, series_b)
        return None

    @staticmethod
    def _op_or_none(series: Optional[pd.Series], op: Callable[[pd.Series], Any]):
        return op(series) if series is not None else None

    def _combine_from_pd_ops(self, other: 'TimeSeries',
                             combine_fn: Callable[[pd.Series, pd.Series], pd.Series]) -> 'TimeSeries':
        """
        Combines this TimeSeries with another one, using the [combine_fn] on the underlying Pandas Series.

        :param other: A second TimeSeries.
        :param combine_fn: An operation with input two Pandas Series and output one Pandas Series.
        :return: A new TimeSeries, with underlying Pandas Series the series obtained with [combine_fn].
        """

        assert self.has_same_time_as(other), 'The two TimeSeries must have the same time index.'

        series = combine_fn(self._series, other.pd_series())
        conf_lo = self._combine_or_none(self._confidence_lo, other.conf_lo_pd_series(), combine_fn)
        conf_hi = self._combine_or_none(self._confidence_hi, other.conf_hi_pd_series(), combine_fn)
        return TimeSeries(series, conf_lo, conf_hi)

    """
    Definition of some useful statistical methods.

    These methods rely on the Pandas implementation.
    """

    def mean(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._series.mean(axis, skipna, level, numeric_only, **kwargs)

    def var(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs) -> float:
        return self._series.var(axis, skipna, level, ddof, numeric_only, **kwargs)

    def std(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs) -> float:
        return self._series.std(axis, skipna, level, ddof, numeric_only, **kwargs)

    def skew(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._series.skew(axis, skipna, level, numeric_only, **kwargs)

    def kurtosis(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._series.kurtosis(axis, skipna, level, numeric_only, **kwargs)

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._series.min(axis, skipna, level, numeric_only, **kwargs)

    def max(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._series.max(axis, skipna, level, numeric_only, **kwargs)

    def sum(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0, **kwargs) -> float:
        return self._series.sum(axis, skipna, level, numeric_only, min_count, **kwargs)

    def median(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs) -> float:
        return self._series.median(axis, skipna, level, numeric_only, **kwargs)

    def autocorr(self, lag=1) -> float:
        return self._series.autocorr(lag)

    def describe(self, percentiles=None, include=None, exclude=None) -> pd.Series:
        return self._series.describe(percentiles, include, exclude)

    """
    Definition of some dunder methods
    """

    def __eq__(self, other):
        if isinstance(other, TimeSeries):
            if not self._series.equals(other.pd_series()):
                return False
            for other_ci, self_ci in zip([other.conf_lo_pd_series(), other.conf_hi_pd_series()],
                                         [self._confidence_lo, self._confidence_hi]):
                if (other_ci is None) ^ (self_ci is None):
                    # only one is None
                    return False
                if self._combine_or_none(self_ci, other_ci, lambda s1, s2: s1.equals(s2)) is False:
                    # Note: we check for "False" explicitly, because None is OK..
                    return False
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self._dataframe)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            new_series = self._series + other
            conf_lo = self._op_or_none(self._confidence_lo, lambda s: s + other)
            conf_hi = self._op_or_none(self._confidence_hi, lambda s: s + other)
            return TimeSeries(new_series, conf_lo, conf_hi)
        elif isinstance(other, TimeSeries):
            return self._combine_from_pd_ops(other, lambda s1, s2: s1 + s2)
        else:
            raise TypeError('unsupported operand type(s) for + or add(): \'{}\' and \'{}\'.'
                            .format(type(self).__name__, type(other).__name__))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            new_series = self._series - other
            conf_lo = self._op_or_none(self._confidence_lo, lambda s: s - other)
            conf_hi = self._op_or_none(self._confidence_hi, lambda s: s - other)
            return TimeSeries(new_series, conf_lo, conf_hi)
        elif isinstance(other, TimeSeries):
            return self._combine_from_pd_ops(other, lambda s1, s2: s1 - s2)
        else:
            raise TypeError('unsupported operand type(s) for - or sub(): \'{}\' and \'{}\'.'
                            .format(type(self).__name__, type(other).__name__))

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_series = self._series * other
            conf_lo = self._op_or_none(self._confidence_lo, lambda s: s * other)
            conf_hi = self._op_or_none(self._confidence_hi, lambda s: s * other)
            return TimeSeries(new_series, conf_lo, conf_hi)
        elif isinstance(other, TimeSeries):
            return self._combine_from_pd_ops(other, lambda s1, s2: s1 * s2)
        else:
            raise TypeError('unsupported operand type(s) for * or mul(): \'{}\' and \'{}\'.'
                            .format(type(self).__name__, type(other).__name__))

    def __rmul__(self, other):
        return self * other

    def __pow__(self, n):
        if isinstance(n, (int, float)):
            if n < 0:
                assert all(self.values() != 0), 'Cannot divide by a TimeSeries with a value 0.'

            new_series = self._series ** float(n)
            conf_lo = self._op_or_none(self._confidence_lo, lambda s: s ** float(n))
            conf_hi = self._op_or_none(self._confidence_hi, lambda s: s ** float(n))
            return TimeSeries(new_series, conf_lo, conf_hi)
        else:
            raise TypeError('unsupported operand type(s) for ** or pow(): \'{}\' and \'{}\'.' \
                            .format(type(self).__name__, type(n).__name__))

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            assert other != 0, 'Cannot divide by 0.'

            new_series = self._series / other
            conf_lo = self._op_or_none(self._confidence_lo, lambda s: s / other)
            conf_hi = self._op_or_none(self._confidence_hi, lambda s: s / other)
            return TimeSeries(new_series, conf_lo, conf_hi)

        elif isinstance(other, TimeSeries):
            assert all(other.values() != 0), 'Cannot divide by a TimeSeries with a value 0.'

            return self._combine_from_pd_ops(other, lambda s1, s2: s1 / s2)
        else:
            raise TypeError('unsupported operand type(s) for / or truediv(): \'{}\' and \'{}\'.' \
                            .format(type(self).__name__, type(other).__name__))

    def __rtruediv__(self, n):
        return n * (self ** (-1))

    def __abs__(self):
        series = abs(self._series)
        conf_lo = self._op_or_none(self._confidence_lo, lambda s: abs(s))
        conf_hi = self._op_or_none(self._confidence_hi, lambda s: abs(s))
        return TimeSeries(series, conf_lo, conf_hi)

    def __neg__(self):
        series = -self._series
        conf_lo = self._op_or_none(self._confidence_lo, lambda s: -s)
        conf_hi = self._op_or_none(self._confidence_hi, lambda s: -s)
        return TimeSeries(series, conf_lo, conf_hi)

    def __contains__(self, item):
        if isinstance(item, pd.Timestamp):
            return item in self._series.index
        return False

    def __round__(self, n=None):
        series = self._series.round(n)
        confidence_lo = self._op_or_none(self._confidence_lo, lambda s: s.round(n))
        confidence_hi = self._op_or_none(self._confidence_hi, lambda s: s.round(n))
        return TimeSeries(series, confidence_lo, confidence_hi)

    # TODO: Ignoring confidence series for now
    def __lt__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            series = self._series < other
        elif isinstance(other, TimeSeries):
            series = self._series < other.pd_series()
        else:
            raise TypeError('unsupported operand type(s) for < : \'{}\' and \'{}\'.'
                            .format(type(self).__name__, type(other).__name__))
        return series  # TODO should we return only the ndarray, the pd series, or our timeseries?

    def __gt__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            series = self._series > other
        elif isinstance(other, TimeSeries):
            series = self._series > other.pd_series()
        else:
            raise TypeError('unsupported operand type(s) for > : \'{}\' and \'{}\'.'
                            .format(type(self).__name__, type(other).__name__))
        return series

    def __le__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            series = self._series <= other
        elif isinstance(other, TimeSeries):
            series = self._series <= other.pd_series()
        else:
            raise TypeError('unsupported operand type(s) for <= : \'{}\' and \'{}\'.'
                            .format(type(self).__name__, type(other).__name__))
        return series

    def __ge__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            series = self._series >= other
        elif isinstance(other, TimeSeries):
            series = self._series >= other.pd_series()
        else:
            raise TypeError('unsupported operand type(s) for >= : \'{}\' and \'{}\'.'
                            .format(type(self).__name__, type(other).__name__))
        return series

    def __str__(self):
        df = pd.DataFrame({'value': self._series})
        if self._confidence_lo is not None:
            df['conf_low'] = self._confidence_lo
        if self._confidence_hi is not None:
            df['conf_high'] = self._confidence_hi
        return str(df) + '\nFreq: {}'.format(self.freq_str())

    def __repr__(self):
        return self.__str__()

    def __copy__(self, deep: bool = True):
        return self.copy(deep=deep)

    def __deepcopy__(self, memodict={}):
        return self.copy(deep=True)

    def __getitem__(self, item):
        # return only main series if nb of values < 3
        if isinstance(item, (int, pd.Timestamp)):
            return self._series[[item]]
        elif isinstance(item, (pd.DatetimeIndex, slice, list, np.ndarray)):
            if isinstance(item, slice):
                # if create a slice with timestamp, convert to indices
                if item.start.__class__ == pd.Timestamp or item.stop.__class__ == pd.Timestamp:
                    istart = None if item.start is None else self.time_index().get_loc(item.start)
                    istop = None if item.stop is None else self.time_index().get_loc(item.stop)
                    item = slice(istart, istop, item.step)
                # cannot reverse order
                if item.indices(len(self))[-1] == -1:
                    raise IndexError("Cannot have a backward TimeSeries")
            # Verify that values in item are really in index to avoid the creation of NaN values
            if isinstance(item, (np.ndarray, pd.DatetimeIndex)):
                check = np.array([elem in self.time_index() for elem in item])
                if not np.all(check):
                    raise IndexError("None of {} in the index".format(item[~check]))
            try:
                return TimeSeries(self._series[item],
                                  self._op_or_none(self._confidence_lo, lambda s: s[item]),
                                  self._op_or_none(self._confidence_hi, lambda s: s[item]))
            except AssertionError:
                # return only main series if nb of values < 3
                return self._series[item]
        else:
            raise IndexError("Input {} of class {} is not a possible key.\n" \
                             "Please use integers, pd.DateTimeIndex, arrays or slice".format(item, item.__class__))

    # def __setattr__(self, *args):
    #     raise TypeError("Can not modify immutable instance.")
    #
    # def __delattr__(self, item):
    #     self.__setattr__()
