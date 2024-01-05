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

    def insert(self, index, item):
        self.pattern.insert(index, item)
        self.pattern = sorted(self.pattern, key=lambda x: x[1])


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


class seqTransltaor(util.univTool):
    def do(self, do_slaves: List[slaveSequence]):
        # Collect all unique times efficiently using a set comprehension.
        times = sorted(
            {time for slave in do_slaves for state, time in slave.pattern.pattern}
        )

        # Iterate through slaves and insert missing times.
        for slave in do_slaves:
            slave_times = [time for state, time in slave.pattern.pattern]
            missing_times = sorted(set(times) - set(slave_times))
            for time in missing_times:
                insertion_index = bisect_left(slave_times, time)

                if insertion_index == 0:
                    state = False
                else:
                    state = slave.pattern.pattern[insertion_index - 1][0]
                slave.pattern.insert(insertion_index, (state, time))
                slave_times.insert(insertion_index, time)

        # State pattern translation
        max_channel = max([slave.channel for slave in do_slaves])
        states = []

        for time in times:
            bit_pattern = bitarray(max_channel + 1)
            bit_pattern.setall(0)  # Set all bits to 0

            for slave in do_slaves:
                state = [state for state, t in slave.pattern.pattern if t == time]
                # print(state)
                if state:
                    bit_pattern[-slave.channel - 1] = state[0]
            # print(bit_pattern)
            states.append(int(bit_pattern.to01(), base=2))

        # Time pattern translation
        times = [int(time / DO_UNIT_TIME) for time in times]

        # Translate to final pattern c_int_array(State,time)
        final_pattern = self.generate_digout_fifo_pattern(states, times)

        return self.conv2C_int(final_pattern)

    def ao(self):
        update_period = float(self.settings.AO.AO_UPDATE_PERIOD)
        duration = float(self.settings.GENERAL.duration)
        # Find empty channels
        tot_channels = set(np.arange(1, 9))
        empty_channels = list(tot_channels - self.AO_chs)
        dump_pattern = slaveSequence.pattern(
            "AO", data=np.zeros(len(self.AO_slaves[0].pattern._data))
        )
        dump_slaves = [
            slaveSequence(
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

        result = self.create_int_values_C(converted)

        return NotImplementedError

    @staticmethod
    def generate_digout_fifo_pattern(values, times):
        k = np.array([0, 1])
        n_values = np.kron(values, k[::-1])  # make [value,0,value,0,...]
        n_times = np.kron(times, k)  # make [0,times,0,times,...]
        pattern = n_values + n_times  # Merge [value, times, value, times, value, times]

        return pattern


class parTranslator(util.univTool):
    def do_ch_pattern(self):
        ch_pattern = bitarray(32)
        ch_pattern.setall(False)
        for ch in self._do_chs:
            ch_pattern[-ch - 1] = True

        return ch_pattern.to01()

    def do_ch_configuration(self):
        # each bit corresponds to 0-7, 8-15, 16-23, 24-31 channels.
        # Find max channel number and set bit to 1
        ch_config = bitarray(4)
        ch_config.setall(False)
        max_ch = max(self._do_chs)

        if max_ch < 8:
            ch_config[-1] = True
        elif max_ch < 16:
            ch_config[-2] = True
        elif max_ch < 24:
            ch_config[-3] = True
        elif max_ch < 32:
            ch_config[-4] = True

        return ch_config.to01()

    def ao(self):
        raise NotImplementedError


class translator(seqTransltaor, parTranslator):
    def __init__(self):
        self.master_sequence = None
        self.adw_params = params.adwinParams()

    def sort_slaves(self, master_sequence: masterSequence):
        """Sorting slave sequences by type

        Returns:
            list: list of sorted master sequences
        """
        slaveSequences = list(master_sequence.slave_sequences.values())
        DO_slaves = [digout for digout in slaveSequences if digout.types == "DO"]
        AO_slaves = [anaout for anaout in slaveSequences if anaout.types == "AO"]
        DI_slaves = [digin for digin in slaveSequences if digin.types == "DI"]
        AI_slaves = [anain for anain in slaveSequences if anain.types == "AI"]

        do_chs = set([int(x.channel) for x in DO_slaves])
        ao_chs = set([int(x.channel) for x in AO_slaves])
        di_chs = set([int(x.channel) for x in DI_slaves])
        ai_chs = set([int(x.channel) for x in AI_slaves])

        self._do_slvs = DO_slaves
        self._ao_slvs = AO_slaves
        self._di_slvs = DI_slaves
        self._ai_slvs = AI_slaves

        self._do_chs = do_chs
        self._ao_chs = ao_chs
        self._di_chs = di_chs
        self._ai_chs = ai_chs

    def translate(self):
        # Sorting
        self.sort_slaves(self.master_sequence)
        # General
        ## Channel configuration
        self.adw_params.generalParams.DIO_CH_CONFIG = self.do_ch_configuration()
        ## Duration
        self.adw_params.generalParams.DURATION = int(
            self.master_sequence.duration / DO_UNIT_TIME
        )
        # * Experiment mode --> Not implemented yet

        # Digital Output
        ##Channel pattern
        self.adw_params.dig_out_params.DO_FIFO_CH_PATTERN = self.do_ch_pattern()
        ## Output sequence pattern
        translated_digout_seq = self.do(self._do_slvs)
        self.adw_params.dig_out_datas.DO_FIFO_PATTERN = translated_digout_seq
        ## FIFO WRITE COUNT = LENGTH OF TRANSLATED DIGOUT SEQUENCE
        self.adw_params.dig_out_params.DO_FIFO_WRITE_COUNT = (
            len(translated_digout_seq) // 2
        )
        ## FIFO WRITE STARTING INDEX = 1
        self.adw_params.dig_out_params.DO_FIFO_WRITE_STARTING_INDEX = 1

        # * Analog Output --> Not implmented yet.


class sequenceManager(dict):
    def __init__(self) -> None:
        super().__init__()

    def __setitem__(self, __value) -> None:
        __key = len(self)
        return super().__setitem__(__key, __value)

    def append(self, master_sequence: masterSequence):
        self.__setitem__(master_sequence)


class deviceManager:
    def __init__(self, boot=True, deviceno=1) -> None:
        self.__adwin = aw.ADwin(DeviceNo=deviceno, useNumpyArrays=True)

        # Boot ADwin-System
        if boot:
            self.boot()
        else:
            print("Adwin not booted, please boot manually")

    def boot(self):
        BTL = self.__adwin.ADwindir + "ADwin" + PROCESSORTYPE + ".btl"
        try:
            self.__adwin.Boot(BTL)
            self.__adwinNo = self.__adwin.DeviceNo
            self.__ProceesorType = self.__adwin.Processor_Type()
        except ValueError:
            raise ValueError("ADwin not connected")

    def load_process(self, mode: Literal["SINGLE", "CONTINUOUS", "SWEEP"]):
        process_no = 1
        if mode == "SINGLE":
            self.__adwin.Load_Process(AD_PROCESS_DIR["SINGLE_MODE"])
        elif mode == "CONTINUOUS":
            self.__adwin.Load_Process(AD_PROCESS_DIR["CONTINUOUS_MODE"])
        elif mode == "SWEEP":
            raise NotImplementedError
        return process_no

    def set_params(self, adwin_params: params.adwinParams, process_no: int):
        num_params = params.paramReferNum().as_dict()
        num_datas = params.dataReferNum().as_dict()
        # For given adwin parameter key, find corresponding adwin parameter number and set value

        for option_name, value in adwin_params.as_dict().items():
            if option_name in num_params:
                # Convert bitstring into base=2 integer
                if isinstance(value, str):
                    value = int(value, base=2)
                self.__adwin.Set_Par(Index=num_params[option_name], Value=value)
            elif option_name in num_datas:
                self.__adwin.SetData_Long(
                    Data=value,
                    DataNo=num_datas[option_name],
                    Startindex=1,
                    Count=len(value),
                )
            else:
                pass

    def start_process(self, process_no: int):
        self.__adwin.Start_Process(process_no)

    def stop_process(self, process_no: int):
        self.__adwin.Stop_Process(process_no)

    def get_status(self, process_no: int):
        status = self.__adwin.Process_Status(process_no)
        match status:
            case 0:
                print("Process is not running")
            case 1:
                print("Process is running")
            case _:
                print("Process is being stopped")
        return

    def get_data(self, data_no: int, start_idx, count):
        return np.array(self.__adwin.GetData_Long(data_no, start_idx, count))

    def get_par(self, par_no: int):
        return self.__adwin.Get_Par(par_no)

    def get_fpar(self, par_no: int):
        return self.__adwin.Get_FPar(par_no)


class validator:
    def __init__():
        return NotImplementedError


class manager:
    def __init__(self, **kwargs):
        boot = kwargs.get("boot", True)
        deviceno = kwargs.get("deviceno", 1)

        self.device_manager = deviceManager(boot, deviceno)
        self.sequence_manager = sequenceManager()
        self.mode = "CONTINUOUS"

    def set_mode(self, mode: Literal["SINGLE", "CONTINUOUS", "SWEEP"]):
        self.mode = mode

    def append(self, master_sequence: masterSequence):
        self.sequence_manager.append(master_sequence)

    def boot(self):
        self.device_manager.boot()

    def translate(self):
        for master_sequence in self.sequence_manager.values():
            self.process_no = self.device_manager.load_process(self.mode)
            trns = translator()
            trns.master_sequence = master_sequence
            trns.translate()
            self.device_manager.set_params(trns.adw_params, self.process_no)

    def start(self):
        # Translate by mode
        self.translate()
        # Start Process
        self.device_manager.start_process(self.process_no)

    def stop(self):
        self.device_manager.stop_process(self.process_no)

    def get_status(self):
        self.device_manager.get_status

    def get_par(self, par_no: int):
        return self.device_manager.get_par(par_no)

    def get_fpar(self, par_no: int):
        return self.device_manager.get_fpar(par_no)

    def get_data(self, data_no: int, start_idx, count):
        return self.device_manager.get_data(data_no, start_idx, count)
