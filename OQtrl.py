import numpy as np
import ADwin as aw
import OQtrl_params as params
import OQtrl_utils as util
from bitarray import bitarray
from bisect import bisect_left
from OQtrl_descriptors import cond_real, OneOf, bit_string
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import reduce
from operator import add
from typing import Literal, List, Dict
from bitarray import bitarray
from matplotlib.pyplot import figure, show

# Path of ADbasic process binary file
AD_PROCESS_DIR = {
    "SINGLE_MODE": NotImplemented,
    "CONTINUOUS_MODE": "./ADbasic_process/Master_Process.TC1",
}

UNIT_TIME = 1e-9  # 1ns
DO_UNIT_TIME = 1e-9 * 10  # 10ns
PROCESSORTYPE = "12"


@dataclass(init=True)
class slaveProperties:
    name: str
    types: Literal["DO", "DI", "AO", "AI"] = OneOf(None, "DO", "DI", "AO", "AI")
    duration: float = None
    update_period: float = None
    channel: cond_real = cond_real(minvalue=0, maxvalue=31, types=int)


@dataclass(init=True)
class masterProperties:
    name: str
    duration: cond_real = cond_real(minvalue=1e-9, types=float)


@dataclass(init=True, repr=True)
class digitalPattern(util.univTool):
    pattern: bit_string = bit_string(None, maxsize=511)

    def __add__(self, other):
        sp = self.str2list(self.pattern)
        op = self.str2list(other.pattern)
        np = self.merge_strlist([sp, op])
        return np

    def __radd__(self, other):
        sp = self.str2list(self.pattern)
        return self.merge_strlist([other, sp])

    def __len__(self):
        return len(self.pattern)


@dataclass(repr=False)
class digitalPatternABS(util.univTool):
    pattern: List[tuple] = field(default_factory=list, init=True)

    def __len__(self):
        return len(self.pattern)

    def __setattr__(self, name, value):
        value = sorted(value, key=lambda x: x[1])

        if name == "pattern":
            if not isinstance(value, list):
                raise TypeError("pattern should be list")
            else:
                if all(isinstance(x, tuple) for x in value):
                    self.__dict__[name] = value
                else:
                    raise TypeError("pattern should be list of tuple")
        else:
            raise AttributeError(f"Invalid attribute {name}")

    def __delattr__(self, name):
        if name == "pattern":
            self.__dict__[name] = []
        else:
            raise AttributeError(f"Invalid attribute {name}")


@dataclass(init=True)
class analogPattern(util.seqTool.patternGenerator):
    pattern: List[float] = None

    def __add__(self, other):
        NotImplementedError

    def __radd__(self, other):
        NotImplementedError

    def __len__(self):
        return len(self.pattern)


class slaveSequence(slaveProperties, util.painter):
    def __init__(
        self,
        name: str,
        types: Literal["DO", "DI", "AO", "AI"],
        duration: float,
        channel: int,
        update_period: float = None,
    ):
        slaveProperties.__init__(
            self,
            name=name,
            types=types,
            duration=duration,
            channel=channel,
            update_period=update_period,
        )
        if self.types == "DO":
            self.pattern = digitalPatternABS()
        elif self.types == "DI":
            self.pattern = digitalPattern()
        elif self.types == "AO" or self.types == "AI":
            self.pattern = analogPattern()

    def __repr__(self) -> str:
        return f"Adwin {self.types} Sequence | Name: {self.name}"

    def __str__(self) -> str:
        return f"{self.pattern.pattern}"

    def __lt__(self, other):
        # Since adwin accept channel pattern as bitarray,
        # we need to sort channel in descending order
        return int(self.channel) > int(other.channel)

    def __len__(self):
        return len(self.pattern)

    def delete(self):
        self.pattern

    def plot(self):
        self.__figure = figure(
            figsize=params.plotParams.FIG_SIZE, dpi=params.plotParams.DPI
        )

        match self.types:
            case "DO":
                self._painter__plot_DO(self.__figure, self)
            case "DI":
                self._painter__plot_DI(self.__figure, self)
            case "AO" | "AI":
                self._painter__plot_analog(self.__figure, self)
            case _:
                raise ValueError(f"Invalid type {self.types}")

        self.__figure.axes[0].set_xlabel(
            "Time (s)", fontsize=params.plotParams.FONT_SIZE
        )
        show()


class masterSequence(masterProperties, util.painter):
    def __init__(self, name: str, duration: float) -> None:
        masterProperties.__init__(self, name=name, duration=duration)
        self.slave_sequences: Dict(slaveSequence) = dict()
        self.plot_params = params.plotParams()

    def __repr__(self) -> str:
        return f" Master Sequence | Name: {self.name}, \n slaves: {[x.name for x in self.slave_sequences.values()]}"

    def __str__(self) -> str:
        return f"{self.slave_sequences}"

    def __getitem__(self, key):
        return self.slave_sequences[key]

    def __setitem__(self, key, value):
        self.slave_sequences[key].pattern.pattern = value

    def set_update_period(self, DI: float, AI: float, AO: float):
        """Set update period for each slave sequence types.
        Args:
            DI (float): detects rising/falling edge every DI seconds
            AI (float): AI samples analog input every AI seconds.
            AO (float): AO outputs analog output every AO seconds.
        """

        self.update_period = {"DI": DI, "AI": AI, "AO": AO}

    def create_slave(
        self,
        types: Literal["DO", "DI", "AO", "AI"] = None,
        name: str = None,
        ch: int = None,
    ):
        """Create new slave sequence to master sequence

        Args:
            type (str): can be "DO", "DI", "AO", "AI"
            ch (int): channel number. If type is "DO" or "DI", ch is digital channel number. If type is "AO" or "AI", ch is analog channel number.
            name (str, optional): Name of sequence. If name is None, name is set to "Sequence #(number)". Defaults to None.

        Raises:
            ValueError: if type is not "DO", "DI", "AO", "AI", raise ValueError
        """
        # Check if name is given
        if name is None:
            # If name is not given, set name to "Sequence #(number)"
            name = f"Sequence # {len(self.slave_sequences)+1}"
        else:
            name = name

        match types:
            case "DO":
                update_period = None
            case "DI":
                update_period = self.update_period["DI"]
            case "AO":
                update_period = self.update_period["AO"]
            case "AI":
                update_period = self.update_period["AI"]
            case _:
                raise ValueError(f"Invalid type {types}")

        # Create sequence object by type
        sequence_obj = slaveSequence(
            name=name,
            duration=self.duration,
            update_period=update_period,
            types=types,
            channel=ch,
        )
        # Append slave sequence object to lists
        self.slave_sequences[f"{name}"] = sequence_obj

    def clear(self) -> None:
        self.slave_sequences.clear()

    def plot(self):
        self.__figure = figure(
            figsize=self.plot_params.FIG_SIZE, dpi=self.plot_params.DPI
        )
        param_dict = self.plot_params.as_dict()

        for sequence in self.slave_sequences.values():
            match sequence.types:
                case "DO":
                    self._painter__plot_DO(self.__figure, sequence, **param_dict)
                case "DI":
                    self._painter__plot_DI(self.__figure, sequence, **param_dict)
                case "AO" | "AI":
                    self._painter__plot_analog(self.__figure, sequence, **param_dict)
                case _:
                    raise ValueError(f"Invalid type {sequence.types}")

            l, b, w, h = param_dict["RECT"]
            new_rect = [l, b - (h + 0.1), w, h]
            param_dict["RECT"] = new_rect

        self.__figure.axes[0].set_xlabel(
            "Time (s)", fontsize=params.plotParams.FONT_SIZE
        )
        show()


@dataclass
class masterSequenceSlots:
    masters_slot: List[masterSequence] = field(default_factory=list)


class seqTransltaor:
    def _do(self, do_slaves: List[slaveSequence]):
        # Collect all unique times efficiently using a set comprehension.
        times = {time for slave in do_slaves for state, time in slave.pattern.pattern}

        # Iterate through slaves and insert missing times.
        for slave in do_slaves:
            slave_times = [time for state, time in slave.pattern.pattern]
            missing_times = set(times) - set(slave_times)

            for time in missing_times:
                insertion_index = bisect_left(slave_times, time)
                state = slave.pattern.pattern[insertion_index - 1][0]
                slave.pattern.pattern.insert(insertion_index, (state, time))

        # State pattern translation
        max_channel = max([slave.channel for slave in do_slaves])
        states = []

        for time in times:
            bit_pattern = bitarray(max_channel + 1)
            bit_pattern.setall(0)  # Set all bits to 0

            for slave in do_slaves:
                state = [state for state, t in slave.pattern.pattern if t == time]
                if state:
                    bit_pattern[-slave.channel - 1] = state[0]
            states.append(int(bit_pattern.to01(), base=2))

        # Time pattern translation
        times = [int(time / DO_UNIT_TIME) for time in times]

        # Create final pattern
        final_pattern = util.seqTool.master.digout_fifo_pattern(states, times)

        return final_pattern

    def _ao(self):
        raise NotImplementedError


class parTranslator:
    def _do(self):
        raise NotImplementedError

    def _ao(self):
        raise NotImplementedError


class translator(masterSequenceSlots, params.adwinParams, seqTransltaor, parTranslator):
    def __init__(
        self,
        mode: Literal["SINGLE", "CONTINUOUS", "SWEEP"],
        master_sequences: List[masterSequence],
        adwin_process=None,
    ):
        super().__init__()
        self._adwinParams__init__()
        self.master_sequences = master_sequences

    def translate(self):
        for master in self.master_sequences:
            sorting = {}
            sorting["DO"].append(
                [x for x in master.slave_sequences.values() if x.types == "DO"]
            )
            sorting["DI"].append(
                [x for x in master.slave_sequences.values() if x.types == "DI"]
            )
            sorting["AO"].append(
                [x for x in master.slave_sequences.values() if x.types == "AO"]
            )
            sorting["AI"].append(
                [x for x in master.slave_sequences.values() if x.types == "AI"]
            )


def translate_AO(self):
    update_period = float(self.settings.AO.AO_UPDATE_PERIOD)
    duration = float(self.settings.GENERAL.duration)
    # Find empty channels
    tot_channels = set(np.arange(1, 9))
    empty_channels = list(tot_channels - self.AO_chs)
    dump_pattern = slaveSequence.pattern(
        "AO", data=np.zeros(len(self.AO_slaves[0].pattern._data))
    )
    dump_slaves = [
        sequence.slaveSequence(
            types="AO",
            duration=duration,
            update_period=update_period,
            channel=ch,
            pattern=dump_pattern,
        )
        for ch in empty_channels
    ]
    self.AO_slaves += dump_slaves
    # Sort by channels [8,7,6,...,2,1]
    self.AO_slaves.sort()
    # Pattern digitize to 16bit
    AOs = [x.pattern._data for x in self.AO_slaves]
    digitized_AOs = [
        np.array(miscs.analog.analog_to_digital(x), dtype=np.int64) for x in AOs
    ]
    bitpattern_AOs = []
    for x in digitized_AOs:
        bitpattern = [f"{val:016b}" for val in x]
        bitpattern_AOs.append(bitpattern)
    print(len(bitpattern_AOs))
    # zip to 32bit pattern, [8,7],[6,5],[4,3],[2,1]
    onetwo = miscs.digital.DO_FIFO.add_string_lists(
        [bitpattern_AOs[-1], bitpattern_AOs[-2]]
    )
    threefour = miscs.digital.DO_FIFO.add_string_lists(
        [bitpattern_AOs[-3], bitpattern_AOs[-4]]
    )
    fivesix = miscs.digital.DO_FIFO.add_string_lists(
        [bitpattern_AOs[-5], bitpattern_AOs[-6]]
    )
    seveneight = miscs.digital.DO_FIFO.add_string_lists(
        [bitpattern_AOs[-7], bitpattern_AOs[-8]]
    )

    # converted
    converted = []
    for i in range(len(onetwo)):
        if i % 4 == 0:
            try:
                converted.append(int(onetwo.pop(0), base=2))
            except TypeError:
                converted.append(0)
        elif i % 4 == 1:
            try:
                converted.append(int(threefour.pop(0, base=2)))
            except TypeError:
                converted.append(0)
        elif i % 4 == 2:
            try:
                converted.append(int(fivesix.pop(0), base=2))
            except TypeError:
                converted.append(0)
        elif i % 4 == 3:
            try:
                converted.append(int(seveneight.pop(0), base=2))
            except TypeError:
                converted.append(0)
    print(converted)
    result = miscs.create_int_values_C(converted)

    self.processed_result["AO"] = result


class sequenceManager(dict):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        return len(self.__master_sequences)

    def __repr__(self):
        return f"{self.__master_sequences}"

    def append(self, master_sequence: masterSequence):
        self.__master_sequences[len(self)] = master_sequence

    def values(self):
        super().values()
        return list(self.values)


class deviceManager:
    def __init__(self, boot=True, deviceno=1) -> None:
        adwin = aw.ADwin(DeviceNo=deviceno, useNumpyArrays=True)
        self.__device_adwin = adwin
        # Boot ADwin-System
        if boot:
            self.boot()
        else:
            print("Adwin not booted, please boot manually")

    def boot(self):
        BTL = self.__device.ADwindir + "ADwin" + PROCESSORTYPE + ".btl"
        try:
            self.__device.Boot(BTL)
            self.__DeviceNo = self.__device.DeviceNo
            self.__ProceesorType = self.__device.Processor_Type()
        except ValueError:
            raise ValueError("ADwin not connected")


class validator:
    pass


class manager(translator, validator):
    def __init__(self, mode: Literal["SINGLE", "CONTINUOUS", "SWEEP"]):
        self.device_manager = deviceManager()
        self.sequence_manager = sequenceManager()
