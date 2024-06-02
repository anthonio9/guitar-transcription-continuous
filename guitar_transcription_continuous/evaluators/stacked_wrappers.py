# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.evaluate import MultipitchEvaluator as _MultipitchEvaluator
from amt_tools.evaluate import PitchListEvaluator as _PitchListEvaluator
from amt_tools.evaluate import NoteEvaluator as _NoteEvaluator

import amt_tools.tools as tools

# Regular imports
import numpy as np

__all__ = [
    'MultipitchEvaluator',
    'PitchListEvaluator',
    'StringAgnosticRMSEEvaluator',
    'StringAgnosticRPAEvaluator',
    'NoteEvaluator',
    'OnsetsEvaluator',
    'OffsetsEvaluator',
]


class MultipitchEvaluator(_MultipitchEvaluator):
    """
    Simple wrapper to support stacked multi pitch estimates and ground-truth.
    """

    def unpack(self, estimated, reference):
        """
        Attempt to unpack and collapse stacked multi pitch data.

        Parameters
        ----------
        estimated : dict
          Dictionary containing stacked multi pitch estimate
        reference : dict
          Dictionary containing stacked multi pitch ground-truth

        Returns
        ----------
        multi_pitch_est : ndarray (F x N)
          Estimated multi pitch data
          F - number of discrete pitches
          N - number of frames
        multi_pitch_ref : ndarray (F x N)
          Ground-truth multi pitch data
          Same dimensions as multi_pitch_est
        """

        # Call the parent function to unpack the stacked data
        stacked_multi_pitch_est, stacked_multi_pitch_ref = super().unpack(estimated, reference)

        # Collapse the stacked multi pitch arrays
        multi_pitch_est = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch_est)
        multi_pitch_ref = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch_ref)

        return multi_pitch_est, multi_pitch_ref


class PitchListEvaluator(_PitchListEvaluator):
    """
    Simple wrapper to support stacked pitch list estimates and ground-truth.
    """

    def unpack(self, estimated, reference):
        """
        Attempt to unpack and collapse stacked pitch list data.

        Parameters
        ----------
        estimated : dict
          Dictionary containing stacked pitch list estimate
        reference : dict
          Dictionary containing stacked pitch list ground-truth

        Returns
        ----------
        pitch_list_est : tuple (times_est, _pitch_list_est)
          Estimated pitch list data
          _pitch_list_est : list of ndarray (T1 x [...])
            Collection of MIDI pitches
          times_est : ndarray (T1)
            Time in seconds associated with each frame
          (T1 - number of observations in estimates (frames))
        pitch_list_ref : tuple (times_ref, _pitch_list_ref)
          Ground-truth pitch list data
          _pitch_list_ref : list of ndarray (T2 x [...])
            Collection of MIDI pitches
          times_ref : ndarray (T2)
            Time in seconds associated with each frame
          (T2 - number of observations in ground-truth (frames))
        """
        # Call the parent function to unpack the stacked data
        stacked_pitch_list_est, stacked_pitch_list_ref = super().unpack(estimated, reference)

        # Collapse the stacked pitch lists
        pitch_list_est = tools.stacked_pitch_list_to_pitch_list(stacked_pitch_list_est)
        pitch_list_ref = tools.stacked_pitch_list_to_pitch_list(stacked_pitch_list_ref)

        return pitch_list_est, pitch_list_ref


class StringAgnosticRMSEEvaluator(PitchListEvaluator):
    """
    Simple evaluator telling the accuracy of predictions according
    to the rule where each row of ground truth is checked for presence
    in the predicted values.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.OCTAVE = 1200
        self.FMIN = 31
        self.THRESHOLD = 50
        self.count = 0
        self.total = 0
        self.results = {}

    def frequency_to_cents(self, frequency):
        """Convert frequency in Hz to cents"""
        if type(frequency) is np.ndarray:
            return self.OCTAVE * np.log2(frequency / self.FMIN)
        else:
            return self.OCTAVE * torch.log2(frequency / self.FMIN)

    def cents_diff(self, a, b):
        """Compute pitch difference in cents"""
        if type(a) is np.ndarray:
            return self.OCTAVE * np.log2(a / b)
        else:
            return self.OCTAVE * torch.log2(a / b)

    def reset_results(self):
        """
        Reset tracked results to empty dictionary.
        """

        super().reset_results()

        self.count = 0
        self.total = 0

    def average_results(self):
        """
        Return the average of the currently tracked results.

        Returns
        ----------
        average : dictionary
          Dictionary with a single value for each metric
        """

        # Average the tracked results
        # average = average_results(self.results)
        # return average
        return np.sqrt((self.total / self.count))


    def evaluate(self, estimated, reference):
        # return only the reference times that exist in 
        est_times = estimated[0]
        est_pitch_list = estimated[1]

        ref_times = reference[0]
        ref_pitch_list = reference[1]

        results = dict()

        for dec in [4, 5, 6]:
            # return indexes of reference for timestamps that exist in estimated
            ref_mask = np.isin(ref_times.round(decimals=dec), est_times.round(decimals=dec))

            # iterate over the mask to get common pitch values
            ref_pitch_list2 = [ref_pitch_list[ind] for (ind, ), mask in np.ndenumerate(ref_mask) if mask]

            # get voiced estimated and reference based on voiced reference
            ref_mask_voiced = np.array([bool(len(pitch_list)) for pitch_list in ref_pitch_list2])

            # estimated pitch list voiced according to reference
            est_pitch_list_voiced = [est_pitch_list[ind] for (ind, ), mask in np.ndenumerate(ref_mask_voiced) if mask]

            # reference pitch list voiced
            ref_pitch_list_voiced = [ref_pitch_list2[ind] for (ind, ), mask in np.ndenumerate(ref_mask_voiced) if mask]

            count_tmp = 0
            total_tmp = 0

            for est, ref in zip(est_pitch_list_voiced, ref_pitch_list_voiced):
                # convert Hz values to cents
                est_cents = self.frequency_to_cents(est)
                ref_cents = self.frequency_to_cents(ref)

                # iterate over every string in the ground truth
                for ref_string in ref_cents:

                    if len(est_cents) == 0:
                        self.count += 1
                        continue

                    # calculate the cents difference between one string ref and all strings estimated
                    difference = self.cents_diff(est_cents.reshape(-1, 1), ref_string.reshape(-1, 1))

                    # find one minimum in each timestamp - find the string with the note
                    difference_min = np.abs(difference).min()

                    count_tmp += 1
                    total_tmp += ((difference_min) ** 2).sum()

            self.count += count_tmp
            self.total += total_tmp

            results[f"rmse-{dec}"] = np.sqrt((self.total / self.count))
        # self.results[self.get_default_key()] 

        # Average the tracked results
        # average = average_results(self.results)
        # return average
        # return np.sqrt((total_tmp / count_tmp))
        return results

    # @staticmethod
    # def get_default_key():
    #     """
    #     Provide the default key to use in the event no key was provided.
    #     """
    #
    #     return "RMSE"


class StringAgnosticRPAEvaluator(StringAgnosticRMSEEvaluator):
    """
    Simple evaluator telling the accuracy of predictions according
    to the rule where each row of ground truth is checked for presence
    in the predicted values.
    """
    def average_results(self):
        """
        Return the average of the currently tracked results.

        Returns
        ----------
        average : dictionary
          Dictionary with a single value for each metric
        """

        # Average the tracked results
        # average = average_results(self.results)
        # return average
        return self.total / self.count

    def evaluate(self, estimated, reference):
        # return only the reference times that exist in 
        est_times = estimated[0]
        est_pitch_list = estimated[1]

        ref_times = reference[0]
        ref_pitch_list = reference[1]

        results = dict()

        for dec in [4, 5, 6]:
            # return indexes of reference for timestamps that exist in estimated
            ref_mask = np.isin(ref_times.round(decimals=4), est_times.round(decimals=4))

            # iterate over the mask to get common pitch values
            ref_pitch_list2 = [ref_pitch_list[ind] for (ind, ), mask in np.ndenumerate(ref_mask) if mask]

            # get voiced estimated and reference based on voiced reference
            ref_mask_voiced = np.array([bool(len(pitch_list)) for pitch_list in ref_pitch_list2])

            # estimated pitch list voiced according to reference
            est_pitch_list_voiced = [est_pitch_list[ind] for (ind, ), mask in np.ndenumerate(ref_mask_voiced) if mask]

            # reference pitch list voiced
            ref_pitch_list_voiced = [ref_pitch_list2[ind] for (ind, ), mask in np.ndenumerate(ref_mask_voiced) if mask]

            count_tmp = 0
            total_tmp = 0

            for est, ref in zip(est_pitch_list_voiced, ref_pitch_list_voiced):
                # convert Hz values to cents
                est_cents = self.frequency_to_cents(est)
                ref_cents = self.frequency_to_cents(ref)

                # iterate over every string in the ground truth
                for ref_string in ref_cents:

                    if len(est_cents) == 0:
                        self.count += 1
                        continue

                    # calculate the cents difference between one string ref and all strings estimated
                    difference = self.cents_diff(est_cents.reshape(-1, 1), ref_string.reshape(-1, 1))

                    # find one minimum in each timestamp - find the string with the note
                    difference_min = np.abs(difference).min()

                    if difference_min < self.THRESHOLD:
                        total_tmp += 1

                    count_tmp += 1

            self.count += count_tmp
            self.total += total_tmp

            results[f"rpa-{dec}"] = self.total / self.count

        return results



class NoteEvaluator(_NoteEvaluator):
    """
    Simple wrapper to support stacked note estimates and ground-truth.
    """

    def unpack(self, estimated, reference):
        """
        Attempt to unpack and collapse stacked note data.

        Parameters
        ----------
        estimated : dict
          Dictionary containing stacked notes estimate
        reference : dict
          Dictionary containing stacked notes ground-truth

        Returns
        ----------
        notes_est : ndarray (K x 3)
          Estimated note intervals and pitches by row
          K - number of estimated notes
        notes_ref : ndarray (L x 3)
          Ground-truth note intervals and pitches by row
          L - number of ground-truth notes
        """

        # Call the parent function to unpack the stacked data
        stacked_notes_est, stacked_notes_ref = super().unpack(estimated, reference)

        # Collapse the stacked notes and convert them to batched representations
        notes_est = tools.notes_to_batched_notes(*tools.stacked_notes_to_notes(stacked_notes_est))
        notes_ref = tools.notes_to_batched_notes(*tools.stacked_notes_to_notes(stacked_notes_ref))

        return notes_est, notes_ref


class OnsetsEvaluator(MultipitchEvaluator):
    """
    Simple wrapper to evaluate stacked onsets estimates and ground-truth.
    """

    @staticmethod
    def get_default_key():
        """
        Default key for onsets activation maps.
        """

        return tools.KEY_ONSETS


class OffsetsEvaluator(MultipitchEvaluator):
    """
    Simple wrapper to evaluate stacked offsets estimates and ground-truth.
    """

    @staticmethod
    def get_default_key():
        """
        Default key for offsets activation maps.
        """

        return tools.KEY_OFFSETS
