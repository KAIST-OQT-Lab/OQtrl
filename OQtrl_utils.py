import matplotlib.pyplot as plt
import numpy as np
import ctypes
import OQtrl_params as params
import bitarray
from dataclasses import dataclass
from functools import reduce
from operator import add
from scipy import signal


class painter:
    def __plot_DO(self, figure, slaveSequence, **plotParams):
        rect = plotParams.get("RECT", params.plotParams().RECT)
        linewidth = plotParams.get("LINEWIDTH", params.plotParams.LINEWIDTH)
        fontsize = plotParams.get("FONTSIZE", params.plotParams.FONT_SIZE)

        pattern = slaveSequence.pattern.pattern
        time = [i[1] for i in pattern]
        value = [i[0] for i in pattern]

        if time[-1] < slaveSequence.duration:
            time.append(slaveSequence.duration)
            value.append(value[-1])

        ax = figure.add_axes(rect)
        ax.step(
            time,
            value,
            where="post",
            color="k",
            linewidth=linewidth,
        )
        name = (
            slaveSequence.name
            + "\n"
            + slaveSequence.types
            + " "
            + str(slaveSequence.channel)
        )
        ax.set_ylabel(name, fontsize=fontsize)
        plt.gca().yaxis.label.set(rotation="horizontal", ha="right")

        self.__figure = figure

    def __plot_DI(self, figure, sequence, **plotParams):
        return NotImplementedError

    def __plot_analog(self, figure, sequence, **plotParams):
        rect = plotParams.get("RECT", params.plotParams().RECT)
        linewidth = plotParams.get("LINEWIDTH", params.plotParams.LINEWIDTH)
        fontsize = plotParams.get("FONTSIZE", params.plotParams.FONT_SIZE)

        # Generate time array for x axis
        time = np.linspace(
            0, float(sequence.duration), len(sequence.pattern._data), endpoint=False
        )
        # add axes to figure
        ax = figure.add_axes(rect)
        # plot
        ax.plot(
            time,
            sequence.pattern._data,
            color="k",
            linewidth=linewidth,
        )
        # y axis label = sequence name
        name = sequence.name + "\n" + sequence.types + " " + str(sequence.channel)
        ax.set_ylabel(name, fontsize=fontsize)
        plt.gca().yaxis.label.set(rotation="horizontal", ha="right")

        self.__figure = figure


class seqTool:
    class patternGenerator:
        @staticmethod
        def __validate_condition(init_volt, end_volt, init_time, end_time, duration):
            # Check if init_volt and end_volt is in range
            if not (init_volt >= -10 and init_volt <= 10):
                raise ValueError("init_volt should be in range [-10,10]")
            if not (end_volt >= -10 and end_volt <= 10):
                raise ValueError("end_volt should be in range [-10,10]")
            # Check if init_time and end_time is in range
            if not (init_time >= 0 and init_time <= duration):
                raise ValueError("init_time should be in range [0,duration]")
            if not (end_time >= 0 and end_time <= duration):
                raise ValueError("end_time should be in range [0,duration]")
            # Check if init_time is smaller than end_time
            if not (init_time < end_time):
                raise ValueError("init_time should be smaller than end_time")
            # Check if init_volt is equal to end_volt
            if init_volt == end_volt:
                raise ValueError("init_volt should be different from end_volt")
            # Check if init_time is equal to end_time
            if init_time == end_time:
                raise ValueError("init_time should be different from end_time")

        def update_pattern(self, pattern, init_time=0, end_time=None):
            duration = self.duration
            update_period = self.update_period

            init_volt, end_volt = pattern[0], pattern[-1]
            self.__validate_condition(
                init_volt, end_volt, init_time, end_time, duration
            )
            total_range = np.zeros(int(duration / update_period) + 1)
            # If init_time is not equal to 0, find index of init_time and insert given pattern into total range
            if init_time != 0:
                init_idx = int(init_time / update_period)
                total_range[init_idx : init_idx + len(pattern)] = pattern
            # If init_time is equal to 0, insert ramp pattern into total range
            else:
                total_range[: len(pattern)] = pattern

            return total_range

        def gen_ramp(self, init_volt, end_volt, init_time=0, end_time=None):
            # make Ramp pattern
            ramp_duration = end_time - init_time
            update_period = self.update_period
            update_bins = int(ramp_duration / update_period) + 1
            update_steps = (end_volt - init_volt) / update_bins
            raw_ramp_pattern = np.arange(init_volt, end_volt, update_steps)

            # update pattern
            ramp_pattern = self.gen_pattern(raw_ramp_pattern, init_time, end_time)

            self.pattern.update(ramp_pattern)

            return self.pattern

        def gen_sin(self, amp, freq, phase=0, offset=0):
            # make sin pattern
            update_period = float(self.update_period)
            duration = float(self.duration)
            t = np.arange(0, duration, update_period)
            sin_pattern = amp * np.sin(2 * np.pi * freq * t + phase) + offset
            # update pattern
            self.pattern.update(sin_pattern)

            return self.pattern

        def gen_square(self, amp, freq, duty=0.5, phase=0, offset=0):
            if self.types == "AO":
                pass
            else:
                raise ValueError(
                    f"Generate square pattern does not support for {self.types}"
                )
            # make square pattern
            update_period = float(self.update_period)
            duration = float(self.duration)
            t = np.arange(0, duration, update_period)
            square_pattern = (
                amp * signal.square(2 * np.pi * freq * t + phase, duty) + offset
            )
            # update pattern
            self.pattern.update(square_pattern)

            return self.pattern

        def gen_sawtooth(self, amp, freq, phase=0, offset=0, width=1):
            # make sawtooth pattern
            update_period = float(self.update_period)
            duration = float(self.duration)
            t = np.arange(0, duration, update_period)
            sawtooth_pattern = (
                amp * signal.sawtooth(2 * np.pi * freq * t + phase, width=width)
                + offset
            )
            # update pattern
            self.pattern.update(sawtooth_pattern)

            return self.pattern

        def gen_gaussian(self, amp, freq, std, offset=0, sym=True):
            # make gaussian pattern
            update_period = float(self.update_period)
            duration = float(self.duration)
            t = np.arange(0, duration, update_period)
            period = 1 / freq
            std = std / update_period
            num_samples = len(np.arange(0, period, update_period))

            gaussian_pattern = (
                amp * signal.windows.gaussian(num_samples, std, sym) + offset
            )

            i = len(gaussian_pattern)
            while i < len(t):
                gaussian_pattern = np.append(gaussian_pattern, gaussian_pattern)
                i = len(gaussian_pattern)

            gaussian_pattern = gaussian_pattern[: len(t)]
            # update pattern
            self.pattern.update(gaussian_pattern)

            return self.pattern

    class slave:
        pass

    class master:
        def sort_slaves(self):
            """Sorting slave sequences by type

            Returns:
                list: list of sorted master sequences
            """
            slaveSequences = list(self.__slaveSequences.values())
            DO_slaves = list()
            AO_slaves = list()
            DI_slaves = list()
            AI_slaves = list()

            # Find slave sequences by type
            for slave in slaveSequences:
                match slave.types:
                    case "DO":
                        DO_slaves.append(slave)
                    case "DI":
                        DI_slaves.append(slave)
                    case "AO":
                        AO_slaves.append(slave)
                    case "AI":
                        AI_slaves.append(slave)
                    case _:
                        raise ValueError(f"Invalid type {slave.types}")

            do_chs = set([int(x.channel) for x in DO_slaves])
            ao_chs = set([int(x.channel) for x in AO_slaves])
            di_chs = set([int(x.channel) for x in DI_slaves])
            ai_chs = set([int(x.channel) for x in AI_slaves])

            self._do = DO_slaves
            self._ao = AO_slaves
            self._di = DI_slaves
            self._ai = AI_slaves

            self._do_chs = do_chs
            self._ao_chs = ao_chs
            self._di_chs = di_chs
            self._ai_chs = ai_chs

        def gen_do_ch_pattern(self, do_chs):
            ch_pattern = bitarray(32)
            ch_pattern.setall(False)
            for ch in do_chs:
                ch_pattern[-ch - 1] = True

            self.params.DO.DO_FIFO_CH_PATTERN = ch_pattern.to01()

        @staticmethod
        def digout_fifo_pattern(values, times):
            k = np.array([0, 1])
            n_values = np.kron(values, k[::-1])  # make [value,0,value,0,...]
            n_times = np.kron(times, k)  # make [0,times,0,times,...]
            pattern = (
                n_values + n_times
            )  # Merge [value, times, value, times, value, times]

            return pattern


class univTool:
    """
    Universal tools
    """

    @staticmethod
    def str2list(string: str):
        """Convert string to list of characters
        Ex) 'Physics' --> ['P','h','y','s','i','c','s']

        Args:
            string (str): string to convert

        Returns:
            list: list of characters
        """
        return [char for char in string]

    @staticmethod
    def merge_strlist(lists: list):
        """Add characters in lists
        Ex) ['1','1','0','0','1','1'] + ['1','0','1','0','1','0'] --> ['11','10','01','00','11','10']
        Args:
            lists (list): list of characters

        Returns:
            _type_: list of added characters
        """
        return reduce(np.char.add, lists).tolist()

    @staticmethod
    def a2d(val: float, bits: int, range: tuple = (-10, 10)) -> int:
        """Convert analog value to digital value

        Args:
            val (float): analog value
            bits (int): number of bits
            range (tuple, optional): (min,max) of analog value. Defaults to (-10,10).

        Returns:
            int: digital value
        """
        return int((val - range[0]) / (range[1] - range[0]) * (2**bits - 1))

    @staticmethod
    def d2a(val: int, bits: int, range: tuple = (-10, 10)) -> float:
        """Convert digital value to analog value

        Args:
            val (int): digital value
            bits (int): number of bits
            range (tuple, optional): (min,max) of analog value. Defaults to (-10,10).

        Returns:
            float: analog value
        """
        return val / (2**bits - 1) * (range[1] - range[0]) + range[0]

    @staticmethod
    def conv2C_int(array) -> ctypes.c_int32:
        """Convert numpy array to ctypes.c_int32

        Args:
            array (numpy.ndarray): numpy array

        Returns:
            ctypes.c_int32: ctypes.c_int32
        """
        return ctypes.c_int32(*array)

    @staticmethod
    def conv2C_float(array) -> ctypes.c_float:
        """Convert numpy array to ctypes.c_float

        Args:
            array (numpy.ndarray): numpy array

        Returns:
            ctypes.c_float: ctypes.c_float
        """
        return ctypes.c_float(*array)
