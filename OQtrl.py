import numpy as np
import ADwin as aw
import OQtrl_settings as OQs
import OQtrl_utils as util
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
    "CONTINUOUS_MODE": "./ADbasic_process/Master_Process.TC1",
}

UNIT_TIME = 1e-9  # 1ns
DO_UNIT_TIME = 1e-9 * 10  # 10ns
PROCESSORTYPE = "12"


@dataclass
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

    # def tolist(self) -> List[int]:
    #     """Make pattern to integer list
    #     Ex) '110011' --> [1,1,0,0,1,1]

    #     Returns:
    #         List[int]: bit pattern integer list
    #     """
    #     return bitarray(self.data).tolist()


@dataclass
class analogPattern(util.seqTool.patternGenerator):
    data: List[float] = None

    def __add__(self, other):
        NotImplementedError

    def __radd__(self, other):
        NotImplementedError

    def __len__(self):
        return len(self.data)


@dataclass
class sequence:
    name: str
    duration: cond_real = cond_real(minvalue=1e-9, types=float)


@dataclass
class slaveSequence(sequence, util.painter):
    types: Literal["DO", "DI", "AO", "AI"] = OneOf(None, "DO", "DI", "AO", "AI")
    update_period: float = None
    channel: cond_real = cond_real(minvalue=0, maxvalue=31, types=int)

    def __post_init__(self):
        if self.update_period is None:
            raise ValueError("update period is not defined")

        self.__settings = OQs.slaveSequenceSetting(
            name=self.name,
            duration=self.duration,
            update_period=self.update_period,
            types=self.types,
            channel=self.channel,
        )

        if self.types == "DO" or self.types == "DI":
            self.pattern = digitalPattern("0")
        elif self.types == "AO" or self.types == "AI":
            self.pattern = analogPattern()

    def __repr__(self) -> str:
        return f"Adwin {self.types} Sequence | Name: {self.name}"

    def __str__(self) -> str:
        return f"{self.pattern}"

    def __lt__(self, other):
        # Since adwin accept channel pattern as bitarray,
        # we need to sort channel in descending order
        return int(self.channel) > int(other.channel)

    def __len__(self):
        return len(self.pattern)

    def update(self, new_pattern):
        self.pattern.update(new_pattern)

    def append(self, new_pattern):
        self.pattern.append(new_pattern)

    def delete(self):
        self.pattern = None

    def plot(self):
        if self.types == "DO" or self.types == "DI":
            self.__plot_digital()
        elif self.types == "AO" or self.types == "AI":
            self.__plot_analog()
        else:
            raise ValueError(f"Invalid type {self.types}")


@dataclass
class masterSequence(sequence, util.painter):
    name: str

    def __post_init__(self) -> None:
        self.slaveSequences: Dict(sequence.slaveSequence) = dict()
        self.params = OQs.masterSequenceSetting(self.name, self.duration)

    def __len__(self) -> int:
        return len(self.duration)

    def __repr__(self) -> str:
        return f" Master Sequence | Name: {self.name}, Duration: {self.duration} \n slaves: {[x.name for x in self.slaveSequences.values()]}"

    def __str__(self) -> str:
        return f"{self.slaveSequences}"

    def __getitem__(self, key):
        return self.slaveSequences[key]

    def append(self, slaveSequence):
        if isinstance(slaveSequence, sequence.slaveSequence):
            self.slaveSequences.append(slaveSequence)
        else:
            if not isinstance(slaveSequence, list | tuple):
                raise TypeError(
                    "slaveSequence should be sequence.slaveSequence or list,tuple of sequence.slaveSequence"
                )
            else:
                if all(isinstance(x, sequence.slaveSequence) for x in slaveSequence):
                    self.slaveSequences += slaveSequence
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
        # Append slave sequence object to lists
        self.slaveSequences[f"{name}"] = sequence_obj

    def clear(self) -> None:
        self.slaveSequences.clear()


@dataclass(repr=False)
class project:
    """This is the second layer of the sequen

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
            self.__data = [x for x in self.__data if x.settings.GENERAL.name != name]
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


class translator:
    def translate_DO(self):
        update_period = self.settings.DO.DO_UPDATE_PERIOD / DO_UNIT_TIME
        tot_channels = set(np.arange(0, max(self._do_chs) + 1))
        empty_channels = list(tot_channels - self._do_chs)
        for ch in empty_channels:
            self.create_slave(types="DO", ch=ch, name=f"Dump_DO{ch}")
            self[f"Dump_DO{ch}"].pattern = "0" * len(self._do[0].pattern)
        self._do.sort()

        do_patterns = [x.pattern for x in self._do]
        do_patterns = np.array([int(x, 2) for x in reduce(add, do_patterns)])
        time_patterns = np.array([int(update_period)]).repeat(len(do_patterns))
        final_do_pattern = util.seqTool.master.digout_fifo_pattern(
            do_patterns, time_patterns
        )

        self.settings.DO.DO_FIFO_WRITE_COUNT = int(len(final_do_pattern) / 2)
        self.__translatedSlaves["DO"] = final_do_pattern

    def translate_AO(self):
        update_period = float(self.settings.AO.AO_UPDATE_PERIOD)
        duration = float(self.settings.GENERAL.duration)
        # Find empty channels
        tot_channels = set(np.arange(1, 9))
        empty_channels = list(tot_channels - self.AO_chs)
        dump_pattern = sequence.pattern(
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


class sequenceManager:
    def __init__(self) -> None:
        self.masterSequences: Dict[sequence.masterSequence] = dict()

    def __setitem__(self, key, value):
        if isinstance(value, sequence.project):
            return self.masterSequences[key] == value
        else:
            raise TypeError("value should be sequence.project")

    def __getitem__(self, key):
        return self.masterSequences[key]


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


class manager(sequenceManager, deviceManager, translator, validator):
    def __init__(self, boot=True, deviceno=1) -> None:
        super().__init__()
        super().__init__(boot=boot, deviceno=deviceno)
        super().__init__()

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
                # self.__device.Start_Process(1)

            case "CONTINUOUS":
                proc_dir = AD_PROCESS_DIR["CONTINUOUS_MODE"]
                idx = kwargs.get("master_idx", 0)
                ad_set, ma_set = proj._return_settings(idx)

                self.__device.Load_Process(proc_dir)
                # Set parameters to adwin
                self.__set_params(ad_set, ma_set, proj[idx])
                self.__device.Set_Processdelay(1, ad_set.GENERAL.PROCESS_DELAY)
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

                # For bitstring, convert bitstring to integer
                if isinstance(value, str):
                    value = int(value, 2)

                # print(par_num, value)
                self.__device.Set_Par(par_num, value)

        for types, values in data.items():
            if values is not None:
                match types:
                    case "DO":
                        dat_num = fin_ad_set._assigned().GLOBAL_DATAS.DO_FIFO_PATTERN
                        # print(par_num, values)
                        self.__device.SetData_Long(
                            values, dat_num, Startindex=1, Count=len(values)
                        )
                    case "AO":
                        dat_num = fin_ad_set._assigned().GLOBAL_DATAS.AO
                        # print(par_num, values)
                        self.__device.SetData_Long(
                            values, dat_num, Startindex=1, Count=len(values)
                        )

    @property
    def deviceno(self):
        return print(self.__DeviceNo)

    @property
    def processor_type(self):
        return print(self.__ProceesorType)

    @property
    def process_delay(self):
        return self.__device.Get_Processdelay(1)

    def show_params_status(
        self, number: int = None, option: str = None, all=True
    ) -> dict or str:
        """return parameters that are now set in ADwin device.

        Args:
            number (int, optional): if specific parameter number is given, return specific parameter. Defaults to None.
            option (str, optional): if specific parameter option is given, return specific parameter. Defaults to None.
            all (bool, optional): if True, return all parameters. Defaults to True.

        Returns:
            dict or str: dictionary of parameters
        """
        # temporal dictionary for parameters
        temp_params_dict = {}

        # If all is True, return all parameters
        if all:
            adwin_params = self.__device.Get_Par_All()
            for option, params in OQs.adwinSetting._assigned().PARAMS.__dict__.items():
                temp_params_dict[option] = adwin_params[params - 1]
            return temp_params_dict

        # If all is False, return specific parameter
        else:
            if number is not None and option is None:
                adwin_params = self.__device.Get_Par(number)
                option = [
                    options
                    for options, params in OQs.adwinSetting._assigned().PARAMS.__dict__.items()
                    if params == number
                ][0]
                return f"{option}:{adwin_params}"

            elif number is None and option is not None:
                par_num = OQs.adwinSetting._assigned().PARAMS.__dict__.get(
                    "DIO_CH_CONFIG"
                )
                return f"option: {self.__device.Get_Par(par_num)}"

    def show_adwin_status(self):
        return self.__device.Process_Status(1)

    def show_process_delay(self):
        return self.__device.Get_Processdelay(1)

    def set_process_delay(self, delay: int):
        self.__device.Set_Processdelay(1, delay)
