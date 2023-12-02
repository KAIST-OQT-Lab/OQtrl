import numpy as np
import ADwin as aw
import OQtrl_settings as OQs
import ctypes
import matplotlib.pyplot as plt
from OQtrl_descriptors import cond_real, OneOf, bit_string
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import reduce
from operator import add
from typing import Literal, List, Dict
from bitarray import bitarray


# Path of ADbasic process binary file
AD_PROCESS_DIR = {
    "SINGLE_MODE": NotImplemented,
    "CONTINUOUS_MODE": './ADbasic_process/Master_Process.TC1'
}

UNIT_TIME = 1e-9  # 1ns
PROCESSORTYPE = "12"


class sequence:
    """
    Sequence classes for adoqt
    """

    class adwinSequence(ABC):
        @abstractmethod
        def append(self):
            pass

    class pattern(adwinSequence):
        types: OneOf("DO", "DI", "AO", "AI")

        def __init__(self, types: Literal["DO", "DI", "AO", "AI"], data=None) -> None:
            self.types = types
            self.update(data)

        def update(self, data):
            if data is None:
                return

            match self.types:
                case "DO":
                    up_pattern = sequence.pattern.DO_pattern(data=data)
                case "DI":
                    up_pattern = sequence.pattern.DI_pattern(data=data)
                case "AO":
                    up_pattern = sequence.pattern.AO_pattern(data=data)
                case "AI":
                    up_pattern = sequence.pattern.AI_pattern(data=data)
                case _:
                    raise ValueError(f"Invalid type {self.types}")

            self.__pattern = up_pattern

        def append(self, pattern):
            NotImplementedError

        def delete(self):
            self.__pattern = None

        @property
        def data(self):
            return self.__pattern

        def __repr__(self) -> str:
            return f"Pattern | Type: {self.types}, Data: {self.__pattern.data}"

        def __str__(self) -> str:
            return f"{self.__pattern.data}"

        @dataclass
        class DO_pattern:
            data: bit_string = bit_string(maxsize=511)
            state: bit_string = bit_string(minsize=1, maxsize=1)

            def __add__(self, other):
                sp = miscs.digital.DO_FIFO.string_to_list(self.data)
                op = miscs.digital.DO_FIFO.string_to_list(other.data)
                np = miscs.digital.DO_FIFO.add_string_lists([sp, op])
                return np

            def __radd__(self, other):
                sp = miscs.digital.DO_FIFO.string_to_list(self.data)
                return miscs.digital.DO_FIFO.add_string_lists([other, sp])

            def tolist(self) -> List[int]:
                """Make pattern to integer list
                Ex) '110011' --> [1,1,0,0,1,1]

                Returns:o
                    List[int]: bit pattern integer list
                """
                return bitarray(self.data).tolist()

        @dataclass
        class DI_pattern:
            data = None

        @dataclass
        class AO_pattern:
            data: List[int] = field(default_factory=list)

        @dataclass
        class AI_pattern:
            data = None

    class slaveSequence(adwinSequence):
        def __init__(self, **kwargs) -> None:
            name = kwargs.get("name", None)
            duration = kwargs.get("duration", None)
            update_period = kwargs.get("update_period", None)
            types = kwargs.get("types", None)
            channel = kwargs.get("channel", None)
            pattern = kwargs.get("pattern", None)

            if update_period is None:
                raise ValueError("update period is not defined")

            self.__settings = OQs.slaveSequenceSetting(
                name=name,
                duration=duration,
                update_period=update_period,
                types=types,
                channel=channel,
            )

            if pattern is None:
                self.__PATTERN = sequence.pattern(types)

            else:
                if not isinstance(pattern, sequence.pattern):
                    raise TypeError("pattern is not sequence.pattern")
                else:
                    self.__PATTERN = pattern

        def __repr__(self) -> str:
            return f"Adwin {self.types} Sequence | Name: {self.name}"

        def __str__(self) -> str:
            return f"{self.__PATTERN.data}"

        def __lt__(self, other):
            # Since adwin accept channel pattern as bitarray,
            # we need to sort channel in descending order
            return int(self.channel) > int(other.channel)

        def update(self, pattern):
            if not isinstance(pattern, sequence.pattern):
                raise TypeError("pattern is not sequence.pattern")
            self.__PATTERN.update(pattern)

        def append(self, pattern):
            self.__PATTERN.append(pattern)

        def delete(self):
            self.__PATTERN = None

        def plot(self, **kwargs):
            figure = kwargs.get("figure", None)

            if figure is None:
                figure = plt.figure(figsize=(5, 3), dpi=300)

            match self.type:
                case "DO":
                    rect = sequence.configuration.plot.INIT_RECT
                    figure = sequence.util.plot_sequence(
                        figure=figure, rect=rect, sequence=self
                    )
                case "DI":
                    return NotImplementedError
                case "AO":
                    return NotImplementedError
                case "AI":
                    return NotImplementedError

            return figure

        def add_note(self, note: str):
            self.__note = note

        @property
        def name(self):
            return f"{self.__settings.GENERAL.name}"

        @property
        def duration(self):
            return f"{self.__settings.GENERAL.duration * UNIT_TIME}"

        @property
        def update_period(self):
            return f"{self.__settings.GENERAL.update_period * UNIT_TIME}"

        @property
        def types(self):
            return f"{self.__settings.GENERAL.types}"

        @property
        def channel(self):
            return f"{self.__settings.GENERAL.channel}"

        @property
        def note(self):
            return self.__note

        @property
        def pattern(self):
            return self.__PATTERN

        @property
        def length(self):
            return int(self.__settings.GENERAL.duration / self.__settings.GENERAL.update_period)

    # Master sequence
    @dataclass(repr=False)
    class masterSequence(adwinSequence):
        def __init__(self, name: str = None, duration: float = 1e-6) -> None:
            # Check if name is given
            if name is None:
                raise ValueError("name is not defined")
            # Check if name is string
            else:
                if not isinstance(name, str):
                    raise TypeError("name should be string")
            
            # Check if duration is given
            if duration is None:
                raise ValueError("duration is not defined")
            else:
                # Check if duration is positive
                if not isinstance(duration, float | int):
                    raise TypeError("duration should be float or integer")
            #We will use duration divided by unit time (1ns).
            duration = duration / UNIT_TIME
            self.__raw_sequences: Dict(sequence.slaveSequence) = dict()
            self.__final_sequences: dict = None
            self.__settings = OQs.masterSequenceSetting(name, duration)
            self.processed = False

        def __len__(self) -> int:
            return len(self.__raw_sequences)

        def __repr__(self) -> str:
            return f" Master Sequence | Name: {self.name}, Duration: {self.duration * UNIT_TIME} \n slaves: {[x.name for x in self.__raw_sequences.values()]}"

        def __str__(self) -> str:
            return f"{self.__raw_sequences}"

        def __getitem__(self, key):
            return self.__raw_sequences[key]

        def append(self, slaveSequence):
            if isinstance(slaveSequence, sequence.slaveSequence):
                self.__raw_sequences.append(slaveSequence)
            else:
                if not isinstance(slaveSequence, list | tuple):
                    raise TypeError(
                        "slaveSequence should be sequence.slaveSequence or list,tuple of sequence.slaveSequence"
                    )
                else:
                    if all(
                        isinstance(x, sequence.slaveSequence) for x in slaveSequence
                    ):
                        self.__raw_sequences += slaveSequence
                    else:
                        raise TypeError(
                            "slaveSequence should be sequence.slaveSequence or list,tuple of sequence.slaveSequence"
                        )
            self.processed = False

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
                name = f"Sequence # {len(self.sequences)+1}"
            else:
                name = name

            match types:
                case "DO":
                    update_period = self.settings.DO.DO_FIFO_UPDATE_PERIOD
                case "DI":
                    update_period = NotImplementedError
                case "AO":
                    update_period = self.settings.AO.AO_UPDATE_PERIOD
                case "AI":
                    update_period = self.settings.AI.AI_UPDATE_PERIOD
                case _:
                    raise ValueError(f"Invalid type {types}")

            # Create sequence object by type
            sequence_obj = sequence.slaveSequence(
                name=name,
                duration=self.duration,
                update_period=update_period,
                types=types,
                channel=ch,
            )
            # Append sequence object to raw sequence
            self.__raw_sequences[f"{name}"] = sequence_obj
            self.processed = False

        def pre_process(self):
            master_seq_list = list(self.__raw_sequences.values())
            pre_processor = _masterSequenceProcessor(master_seq_list, self.settings)
            pre_processor.fianlize()

            self.__settings = pre_processor.return_settings()
            self.__final_sequences = pre_processor.result

            self.processed = True

        def plot(self):
            # Plotting slave sequences
            bob = painter(self.__raw_sequences.values())
            bob.plot()

        @property
        def name(self):
            return self.settings.GENERAL.name

        @property
        def duration(self):
            return self.settings.GENERAL.duration

        @property
        def settings(self):
            return self.__settings

        def _return_final_sequences(self):
            return self.__final_sequences

    @dataclass(repr=False)
    class project(adwinSequence):
        """This is the second layer of the sequence, a collection class of multiple Master Sequences.

        Raises:
            TypeError: when appending masterSequences, if masterSequences is not sequence.masterSequence or list,tuple or dictionary of sequence.masterSequence, raise TypeError
            ValueError: when deleting masterSequences, if name or index is not given, raise ValueError

        Returns:
            project: project class
        """

        name: str = field(default=None, init=True, repr=True)
        order: int = cond_real(types=int)

        def __post_init__(self) -> None:
            self.__data: Dict(sequence.masterSequence) = dict()
            self.__data_settings: Dict(OQs.masterSequenceSetting) = dict()
            self.settings = OQs.adwinSetting()

        def __repr__(self) -> str:
            return f"Project | Name: {self.name}, Number of Master Sequences: {len(self.__data)}"

        def __str__(self) -> str:
            return f"{self.__data}"

        def __getitem__(self, key):
            return self.__data[key]

        def append(self, masterSequences, order: int = None) -> None:
            # If order is None, append master sequence to the end of the dictionary
            if order is None:
                # If dictionary is empty, order is 0
                if len(self.__data) == 0:
                    order = 0
                # If dictionary is not empty, order is the last order + 1
                else:
                    order = len(self.__data) + 1
            else:
                # Check if order already exists
                if order in self.__data.keys():
                    raise ValueError(f"order {order} already exists")
                else:
                    order = order

            # Check if masterSequences is processed
            if not masterSequences.processed:
                # if not, pre_process masterSequences
                masterSequences.pre_process()

            # Check if masterSequences is sequence.masterSequence
            if isinstance(masterSequences, sequence.masterSequence):
                if not masterSequences.processed:
                    masterSequences.pre_process()
                else:
                    pass
            # Check if masterSequences is list,tuple or dictionary of sequence.masterSequence
            else:
                # Check if masterSequences is list,tuple or dictionary
                if not isinstance(masterSequences, list | tuple | dict):
                    raise TypeError(
                        "masterSequences should be sequence.masterSequence or list,tuple or dictionary of sequence.masterSequence"
                    )
                else:
                    NotImplementedError

            self.__data[order] = masterSequences._return_final_sequences()
            self.__data_settings[order] = masterSequences.settings
            # Ordering dictionary key
            self.__data = miscs.ordering_dict_key(self.__data)

        # Swap method for master sequence
        def swap(self, key_1: int, key_2: int) -> None:
            order = (key_1, key_2)
            self.__data = miscs.swap_dict_value(dictionary=self.__data, keys=order)
            self.__data_settings = miscs.swap_dict_value(
                dictionary=self.__data_settings, keys=order
            )

        def delete(self, **kwargs) -> None:
            """Delete master sequence by name or index

            Raises:
                ValueError: if name or index is not given, raise ValueError
            """
            # Check if name or index is given
            name: str = kwargs.get("name", None)
            index: int = kwargs.get("index", None)

            # If name is given, delete master sequence by name
            if name is not None and isinstance(name, str):
                self.__data = [
                    x for x in self.__data if x.settings.GENERAL.name != name
                ]
                self.__data = miscs.ordering_dict_key(self.__data)
            # If index is given, delete master sequence by index
            elif index is not None:
                del self.__data[index]
                self.__data = miscs.ordering_dict_key(self.__data)
            # If name or index is not given, raise ValueError
            else:
                raise ValueError("name or index should be given")

        def _return_settings(self, idx):
            return self.settings, self.__data_settings[idx]

        @property
        def mode(self):
            return self.settings.GENERAL.EXPERIMENT_MODE


class _masterSequenceProcessor:
    def __init__(
        self,
        slaveSequences: List[sequence.slaveSequence] = None,
        settings: OQs.masterSequenceSetting = None,
    ) -> None:
        # Check if masterSequences is list of sequence.masterSequence
        if slaveSequences is None:
            raise ValueError("Requries slave sequences")
        else:
            if not isinstance(slaveSequences, list | tuple):
                raise TypeError(
                    "masterSequences should be list or tuple of sequence.masterSequence"
                )
            else:
                if all(isinstance(x, sequence.slaveSequence) for x in slaveSequences):
                    self.slaveSequences = slaveSequences
                else:
                    raise TypeError(
                        "masterSequences should be list or tuple of sequence.masterSequence"
                    )
        # Check if settings is OQs.masterSequence_SETTINGS
        if settings is None:
            raise ValueError("Requires master sequence settings")
        elif not isinstance(settings, OQs.masterSequenceSetting):
            raise TypeError("settings should be OQs.masterSequence_SETTINGS")
        else:
            self.settings = settings

        self.processed_result = {"DO": None, "AO": None, "DI": None, "AI": None}
        (
            self.DO_slaves,
            self.AO_slaves,
            self.DI_slaves,
            self.AI_slaves,
        ) = miscs.slaves_seq_finder(self.slaveSequences)
        self.parameters = dict()

        # Find Activate Channels
        self.DO_chs = set([int(x.channel) for x in self.DO_slaves])
        self.AO_chs = set([int(x.channel) for x in self.AO_slaves])
        self.DI_chs = set([int(x.channel) for x in self.DI_slaves])
        self.AI_chs = set([int(x.channel) for x in self.AI_slaves])

    def fianlize(self):
        self.validate()
        self.merge_slaves()
        self.generate_ch_pattern()

    def validate(self):
        return None

    def merge_slaves(self):
        self.__merge_slaves_DO()
        self.__merge_slaves_AO()
        self.__merge_slaves_DI()
        self.__merge_slaves_AI()

    def __merge_slaves_DO(self):
        update_period = self.settings.DO.DO_FIFO_UPDATE_PERIOD
        duration = self.settings.GENERAL.duration
        # Find empty channels between sorted DIO sequences
        tot_channels = set(np.arange(0, max(self.DO_chs)+1))
        # Find empty channels
        empty_channels = list(tot_channels - self.DO_chs)
        # Create dump slaves for empty channels
        dump_slaves = [
            sequence.pattern(
                types="DO",
                duration=duration,
                update_period=update_period,
                channel=ch,
            )
            for ch in empty_channels
        ]
        # Add empty channels
        self.DO_slaves += dump_slaves
        # Sort DO sequences by channel
        self.DO_slaves.sort()

        # DO_Patterns
        DO_patterns = [x.pattern.data for x in self.DO_slaves]
        # Adding patterns
        DO_signals = np.array([int(x,2) for x in reduce(add, DO_patterns)])
        # Correspoding update period time pattern
        time_pattern = np.array([int(update_period)]).repeat(len(DO_signals))
        # Generate DO FIFO pattern (pattern, time, pattern, time ...)
        Final_DIO_pattern = miscs.digital.DO_FIFO.pattern_gen(DO_signals, time_pattern)

        self.settings.DO.DO_FIFO_WRITE_COUNT = int(len(Final_DIO_pattern) / 2)
        self.processed_result["DO"] = miscs.create_int_values_C(Final_DIO_pattern)

    def generate_ch_pattern(self):
        # Generate DO Channel Pattern (DO_FIFO_CH_PATTERN)
        ch_pattern = bitarray(32)
        ch_pattern.setall(0)

        for ch in self.DO_chs:
            ch_pattern[-ch - 1] = 1

        self.settings.DO.DO_FIFO_CH_PATTERN = ch_pattern.to01()

    def __merge_slaves_AO(self):
        return None

    def __merge_slaves_DI(self):
        return None

    def __merge_slaves_AI(self):
        return None

    def return_settings(self):
        return self.settings

    @property
    def result(self):
        return self.processed_result


class manager:
    def __init__(self, boot=True, deviceno=1) -> None:
        adwin = aw.ADwin(DeviceNo=deviceno, useNumpyArrays=True)
        self.__device = adwin
        self.__boot_status = boot
        self.__projects = dict()
        # Boot ADwin-System
        if boot:
            self.boot()
            self.__boot_status = True
        else:
            print("Adwin not booted, please boot manually")

    def __setitem__(self, key, value):
        if isinstance(value, sequence.project):
            return self.__projects[key] == value
        else:
            raise TypeError("value should be sequence.project")

    def __getitem__(self, key):
        return self.__projects[key]

    def boot(self):
        BTL = self.__device.ADwindir + "ADwin" + PROCESSORTYPE + ".btl"
        try:
            self.__device.Boot(BTL)
            self.__DeviceNo = self.__device.DeviceNo
            self.__ProceesorType = self.__device.Processor_Type()
        except ValueError:
            raise ValueError("ADwin not connected")

    def create_project(self, name: str = None, order: int = None):
        """Create new project to manager

        Args:
            name (str, optional): name of project. Defaults to None.
            order (int, optional): order for projects.

        Returns:
            sequence.project: project class
        """

        project = sequence.project(name=name, order=order)
        self.__projects[f"{name}"] = project

    def start(self, project: str = None, **kwargs):
        # If ADwin is not booted, boot ADwin
        if not self.__boot_status:
            self.boot()

        # Check if proj_key is given
        if project is None:
            raise ValueError("proj_key is not defined")
        else:
            # Check if proj_key is valid
            if project not in self.__projects.keys():
                raise ValueError(f"Project should be one of {self.__projects.keys()}")

        proj = self.__projects[project]
        mode = proj.mode

        match mode:
            # SINGLE MODE NOT IMPLEMENTED in this version
            case "SINGLE":
                if kwargs.get("master_idx", None) is None:
                    raise ValueError("master_idx should be given for SINGLE mode")
                else:
                    master_idx = kwargs.get("master_idx", None)

                proc_dir = AD_PROCESS_DIR["SINGLE_MODE"]
                self.__device.Load_Process(proc_dir)
                #self.__device.Start_Process(1)

            case "CONTINUOUS":
                proc_dir = AD_PROCESS_DIR["CONTINUOUS_MODE"]
                idx = kwargs.get("master_idx", 0)
                ad_set, ma_set = proj._return_settings(idx)

                self.__device.Load_Process(proc_dir)
                # Set parameters to adwin
                self.__set_params(ad_set, ma_set, proj[idx])

                self.__device.Start_Process(1)

            # SERIES MODE NOT IMPLEMENTED in this version
            case "SERIES":
                proc_dir = AD_PROCESS_DIR["SERIES_MODE"]
            case _:
                raise ValueError(f"Invalid mode {mode}")

    def stop(self):
        self.__device.Stop_Process(1)

    def __set_params(self, ad_set, ma_set, data):
        assigner = OQs.settingAssigner(ad_set, ma_set)
        assigner.assign()
        fin_ad_set = assigner.adw

        for option, value in fin_ad_set.show_options().items():
            if option in fin_ad_set.show_params().keys():
                par_num = fin_ad_set.show_params()[option]
                
                #For bitstring, convert bitstring to integer
                if isinstance(value, str):
                    value = int(value, 2)
            
                #print(par_num, value)
                self.__device.Set_Par(par_num,value)

        for types, values in data.items():
            if values is not None:
                match types:
                    case "DO":
                        dat_num = fin_ad_set._assigned().GLOBAL_DATAS.DO_FIFO_PATTERN
                        # print(par_num, values)
                        self.__device.SetData_Long(values,dat_num, Startindex=1, Count=len(values))
                    case "AO":
                        dat_num = fin_ad_set._assigned().GLOBAL_DATAS.AO_PATTERN
                        # print(par_num, values)
                        self.__device.SetData_Long(values,dat_num, Startindex=1, Count=len(values))

    @property
    def deviceno(self):
        return print(self.__DeviceNo)

    @property
    def processor_type(self):
        return print(self.__ProceesorType)
    
    @property
    def process_delay(self):
        return self.__device.Get_Processdelay(1)

    def show_params_status(self,number:int=None, option:str=None, all=True)->dict or str:
        """return parameters that are now set in ADwin device.

        Args:
            number (int, optional): if specific parameter number is given, return specific parameter. Defaults to None.
            option (str, optional): if specific parameter option is given, return specific parameter. Defaults to None.
            all (bool, optional): if True, return all parameters. Defaults to True.

        Returns:
            dict or str: dictionary of parameters
        """
        #temporal dictionary for parameters
        temp_params_dict = {}

        #If all is True, return all parameters
        if all:
            adwin_params = self.__device.Get_Par_All()
            for option, params in OQs.adwinSetting._assigned().PARAMS.__dict__.items():
                temp_params_dict[option] = adwin_params[params-1]
            return temp_params_dict
        
        #If all is False, return specific parameter
        else:
            if number is not None and option is None:
                adwin_params = self.__device.Get_Par(number)
                option = [options for options, params in OQs.adwinSetting._assigned().PARAMS.__dict__.items() if params == number][0]
                return f'{option}:,{adwin_params}'

            elif number is None and option is not None:
                par_num = OQs.adwinSetting._assigned().PARAMS.__dict__.get('DIO_CH_CONFIG')
                return f'option: {self.__device.Get_Par(par_num)}'

    def show_adwin_status(self):
        return self.__device.Process_Status(1)        
    
class painter:
    def __init__(self, sequences: sequence.masterSequence) -> None:
        self.sequences = sequences
        self.configuration = painter.plot_configuration()
        self.figure = None

    def plot(self):
        self.figure = plt.figure(figsize=(5, 3), dpi=500)
        rect = self.configuration.INIT_RECT

        for sequence in self.sequences:
            match sequence.types:
                case "DO":
                    self.plot_DO(self.figure, sequence, rect)
                case "DI":
                    self.plot_DI(self.figure, sequence, rect)
                case "AO":
                    self.plot_AO(self.figure, sequence, rect)
                case "AI":
                    self.plot_AI(self.figure, sequence, rect)
                case _:
                    raise ValueError(f"Invalid type {sequence.types}")

            l, b, w, h = rect
            rect = [l, b - (h + 0.1), w, h]

    def plot_DO(self, figure, sequence, rect, color: str = "k"):
        if rect is None:
            rect = self.configuration.INIT_RECT
        # Generate time array for x axis
        time = np.linspace(0, float(sequence.duration) * UNIT_TIME, sequence.length)
        # add axes to figure
        ax = figure.add_axes(rect)
        # plot step function
        ax.step(
            time,
            sequence.pattern.data.tolist(),
            where="post",
            color=color,
            linewidth=self.configuration.LINEWIDTH,
        )
        # y axis label = sequence name
        name = sequence.name + "\n" + sequence.types + " " + str(sequence.channel)
        ax.set_ylabel(name, fontsize=self.configuration.FONT_SIZE)
        plt.gca().yaxis.label.set(rotation="horizontal", ha="right")

        self.figure = figure

    def plot_DI(self, figure, sequence, rect, color: str = "k"):
        pass

    def plot_AO(self, figure, sequence, rect, color: str = "k"):
        pass

    def plot_AI(self, figure, sequence, rect, color: str = "k"):
        pass

    @dataclass(frozen=True)
    class plot_configuration:
        LINEWIDTH = 2
        INIT_RECT = [0, 0, 1.2, 0.4]  # left, bottom, width, height
        FONT_SIZE = 10  # font size for axis label


class miscs:
    @staticmethod
    def slaves_seq_finder(slaveSequences: List[sequence.slaveSequence], **kwagrs):
        """Find master sequences by type

        Args:
            masterSequences (Dict[int,sequence.masterSequence]): _description_

        Raises:
            TypeError: if masterSequences is not Dict[int,sequence.masterSequence], raise TypeError

        Returns:
            list: list of sorted master sequences
        """
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

        return DO_slaves, AO_slaves, DI_slaves, AI_slaves

    @staticmethod
    def ordering_dict_key(dictionary: dict) -> dict:
        """For a given dictionary, make the keys in consecutive ascending order.
        For example,
        Args:
            dictionary (dict): dictionary to order

        Returns:
            dict: ordered dictionary
        """
        copied_dict: dict = deepcopy(dictionary)
        sorted_keys = sorted(copied_dict.keys())
        current_key = sorted_keys[0]

        for key in sorted_keys[1:]:
            if key != current_key + 1:
                copied_dict[current_key + 1] = copied_dict.pop(key)
            current_key += 1

        return copied_dict

    @staticmethod
    def swap_dict_value(dictionary: dict, *keys: List[int]) -> dict:
        """Swap the values of the original and moving keys in the dictionary.
        *keys should be a tuple of two keys to swap.

        For example, swap_dict_value( A, 1, 3) for A = {1:'first',2:'second',3:'third'} returns
        --> A' = {1:'third',2:'second',3:'first'}

        Args:
            dictionary (dict): dictionary to swap

        Raises:
            ValueError: if the number of keys is not 2, raise ValueError
            KeyError: if the original or moving key does not exist in the dictionary, raise KeyError

        Returns:
            dict: swapped dictionary
        """
        if len(keys) != 2:
            raise ValueError("Invalid number of keys")

        key_1, key_2 = keys

        # Get the values corresponding to the original and moving indices
        original_value = dictionary.get(key_1)
        moving_value = dictionary.get(key_2)

        # Check if both indices exist in the dictionary
        if original_value is not None and moving_value is not None:
            # Swap the values in the dictionary
            dictionary[key_1] = moving_value
            dictionary[key_2] = original_value
        else:
            raise KeyError("Invalid key")

        return dictionary

    @staticmethod
    def create_int_values_C(array) -> ctypes.c_int32:
        """creat integer C type array from python array

        Args:
            array: array to convert

        Returns:
            ctypes.c_int32: converted array
        """
        datatype = ctypes.c_int32 * len(array)
        return datatype(*array)

    class analog:
        @staticmethod
        def analog_to_digital(x, bit=16, range=(-10, 10)):
            range_min, range_max = range

            ans = (x + range_max) * 2**bit / (range_max - range_min)

            return ans

        @staticmethod
        def digital_to_analog(x, bit=16, range=(-10, 10)):
            range_min, range_max = range

            ans = range_min + (x * (range_max - range_min) / 2**bit)

            return ans

    class digital:
        class DO_FIFO:
            @staticmethod
            def bit_array_gen(length: int) -> bitarray:
                """Generate initialized bit array with given length

                Args:
                    length (int): length of bit array

                Returns:
                    bitarray: 0 initialized bit array
                """
                bit_arr = bitarray(length)
                bit_arr.setall(0)

                return bit_arr

            @staticmethod
            def bit_array_edit(bitarray, channel: int, state: bool):
                length = len(bitarray)
                index = length - channel
                bitarray[index] = state

                return bitarray

            @staticmethod
            def pattern_gen(values, times):
                """For given values and times arrays, Merge [value1, times1, value2, times2, value, times ...]

                Args:
                    values (arrayLike): bit array pattern [0000,0100,1010,...]
                    times (arrayLike): relative or absolute time for values

                Returns:
                    Array: Pattern array for ADWin device [value1, times1, value2, times2, value, times ...]
                """
                k = np.array([0, 1])
                n_values = np.kron(values, k[::-1])  # make [value,0,value,0,...]
                n_times = np.kron(times, k)  # make [0,times,0,times,...]
                pattern = (
                    n_values + n_times
                )  # Merge [value, times, value, times, value, times]

                return pattern

            @staticmethod
            def string_to_list(string: str):
                """Convert string to list
                Ex) 'Physics' --> ['P','h','y','s','i','c','s']

                Args:
                    string (str): string to convert

                Returns:
                    list: list of characters
                """
                return [char for char in string]

            @staticmethod
            def add_string_lists(lists: list):
                """Add characters in lists
                Ex) ['1','1','0','0','1','1'] + ['1','0','1','0','1','0'] --> ['11','10','01','00','11','10']
                Args:
                    lists (list): list of characters

                Returns:
                    _type_: list of added characters
                """
                return reduce(np.char.add, lists).tolist()
