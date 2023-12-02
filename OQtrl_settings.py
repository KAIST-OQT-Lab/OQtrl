from typing import Literal
from dataclasses import dataclass, field, asdict
from OQtrl_descriptors import cond_real, bit_string, OneOf

UNIT_TIME = 1e-9  # 1ns

class adwinLimits:
    @dataclass(frozen=True)
    class _MIN_AO:
        AO_MIN_base10: int = -10
        AO_MAX_base10: int = 10

    @dataclass(frozen=True)
    class _MIN_AI_BURST_CLOCK_RATE:
        AI_AVG_MODE_0: int = 25
        AI_AVG_MODE_1: int = 67
        AI_AVG_MODE_2: int = 154
        AI_AVG_MODE_3: int = 313
        AI_AVG_MODE_4: int = 645
        AI_AVG_MODE_5: int = 1333


@dataclass(init=False)
class adwinSetting:
    """ADWIN settings class
    Types: GENERAL, DO, DI, AO, AI
    Options:
        Gneral: DIO_CH_CONFIG, PROCESS_DELAY, EXPERIMENT_MODE
        DO: DO_FIFO_CH_PATTERN, DO_FIFO_WRITE_COUNT, DO_FIFO_WRITE_STARTING_INDEX
        DI: None
        AO: None
        AI: AI_AVG_MODE, AI_BURST_CHANNELS, AI_BURST_BUFFER_SIZE, AI_BURST_CLOCK_RATE, AI_BURST_TRIGGER_MODE
    Values :
        Corresponding values for each option
    """

    def __init__(self) -> None:
        self.GENERAL = self._general()
        self.DO = self._DO()
        self.DI = self._DI()
        self.AO = self._AO()
        self.AI = self._AI()

    def todict(self) -> dict:
        """return all the options and their values in a dictionary"""
        return {
            types: asdict(self.__dict__[types])
            for types in self.__dict__.keys()
            if types != "ASSIGNED"
        }

    def isnone(self) -> dict:
        """if any of the options is None, return the type of the option and the option name

        types: one of 'GENERAL','DO','DI','AO','AI'
        option: one of the option name in the corresponding type
            ex) GENERAL: 'DIO_CH_CONFIG', 'PROCESS_DELAY', 'EXPERIMENT_MODE'

        Returns:
            dict: {types: option}
        """
        none_options = {}
        for types, options in self.todict().items():
            for option, value in options.items():
                if value is None:
                    none_options[types] = option
        return none_options

    def show_options(self) -> dict:
        """return all the options and their values in a dictionary

        Returns:
            dict: {option: value}
        """
        total_options = {}

        for _, options in self.todict().items():
            for option, value in options.items():
                total_options[option] = value
        return total_options

    def show_params(self) -> dict:
        """return all the options and their assigend parameter values in a dictionary

        Returns:
            dict: {option: parameter number}
        """
        total_params = {}

        for _, v in adwinSetting._assigned().dict().items():
            for option, par_num in v.items():
                total_params[option] = par_num

        return total_params

    @dataclass
    class _general:
        DIO_CH_CONFIG: bit_string = bit_string("11", maxsize=4)
        PROCESS_DELAY: cond_real = cond_real(
            1000, minvalue=1, types=int
        )  # Process delay in unit of ns
        EXPERIMENT_MODE: Literal["SINGLE", "CONTINUOUS", "SERIES"] = OneOf(
            "CONTINUOUS", "SINGLE", "CONTINUOUS", "SERIES"
        )

    @dataclass
    class _DO:
        DO_FIFO_CH_PATTERN: bit_string = bit_string(
            maxsize=32
        )  # This value will be assigned from the master sequence
        DO_FIFO_WRITE_COUNT: int = cond_real(
            types=int
        )  # This value will be assigned from the master sequence unless user specifies
        DO_FIFO_WRITE_STARTING_INDEX: cond_real = cond_real(1, types=int, minvalue=0)

    @dataclass
    class _DI:
        pass

    @dataclass
    class _AO:
        AO_UPDATE_PERIOD: cond_real = cond_real(
            3000, minvalue=3000, types=int
        )  # This value will be assigned from the master sequence

    @dataclass
    class _AI:
        AI_AVG_MODE: Literal[0, 1, 2, 3, 4, 5] = OneOf(0, 0, 1, 2, 3, 4, 5)
        AI_BURST_CHANNELS: int = OneOf(255, 1, 3, 15, 255)
        AI_BURST_BUFFER_SIZE: int = 20000
        AI_BURST_CLOCK_RATE: int = None
        AI_BURST_TRIGGER_MODE: bit_string = bit_string("010", minsize=2, maxsize=3)

    @dataclass
    class _assigned:
        def __init__(self) -> None:
            self.GLOBAL_DATAS = self._ASSIGNED_GLOBAL_DATAS()
            self.PARAMS = self._ASSIGNED_PARAMS()

        def dict(self):
            return {
                types: asdict(self.__dict__[types]) for types in self.__dict__.keys()
            }

        @dataclass(frozen=True)
        class _ASSIGNED_GLOBAL_DATAS:
            DO_FIFO_PATTERN: int = 30
            AI_CH1_PATTERN: int = 40
            AI_CH2_PATTERN: int = 41
            AI_CH3_PATTERN: int = 42
            AI_CH4_PATTERN: int = 43
            AI_CH5_PATTERN: int = 44
            AI_CH6_PATTERN: int = 45
            AI_CH7_PATTERN: int = 46
            AI_CH8_PATTERN: int = 47
            AO_PATTERN: int = 50

        @dataclass(frozen=True)
        class _ASSIGNED_PARAMS:
            DIO_CH_CONFIG: int = 1
            DO_FIFO_CH_PATTERN: int = 31
            DO_FIFO_WRITE_COUNT: int = 32
            DO_FIFO_WRITE_STARTING_INDEX: int = 33
            AI_AVG_MODE: int = 40
            AI_BURST_CHANNELS: int = 41
            AI_BURST_BUFFER_SIZE: int = 42
            AI_BURST_CLOCK_RATE: int = 43
            AI_BURST_TRIGGER_MODE: int = 44
            AO_UPDATE_PERIOD: int = 50


class masterSequenceSetting:
    def __init__(self, name, duration) -> None:
        self.GENERAL = masterSequenceSetting.general(name=name, duration=duration)
        self.DO = masterSequenceSetting.DigOut()
        self.DI = masterSequenceSetting.DigIn()
        self.AO = masterSequenceSetting.AnaOut()
        self.AI = masterSequenceSetting.AnaIn()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.GENERAL.name}, duration={self.GENERAL.duration})"

    def todict(self) -> dict:
        return {
            types: asdict(self.__dict__[types])
            for types in self.__dict__.keys()
            if types != "ASSIGNED"
        }

    def isnone(self) -> dict:
        none_options = {}
        for types, options in self.todict().items():
            for option, value in options.items():
                if value is None:
                    none_options[types] = option
        return none_options

    def show_options(self) -> dict:
        total_options = {}

        for _, options in self.todict().items():
            for option, value in options.items():
                total_options[option] = value
        return total_options

    def set_update_period(self, **kwagrs):
        """For given types, set the update period to the given value in seconds
        """
        
        DO_up = kwagrs.get("DO", None)
        DI_up = kwagrs.get("DI", None)
        AO_up = kwagrs.get("AO", None)
        AI_up = kwagrs.get("AI", None)

        if DO_up is not None:
            self.DO.DO_FIFO_UPDATE_PERIOD = int(DO_up/UNIT_TIME)
        if DI_up is not None:
            self.DI.DI_UPDATE_PERIOD = DI_up
        if AO_up is not None:
            self.AO.AO_UPDATE_PERIOD = AO_up
        if AI_up is not None:
            self.AI.AI_UPDATE_PERIOD = AI_up

    @dataclass
    class general:
        name: str = field(init=True)
        duration: float = field(init=True)
        PROCESS_DELAY: cond_real = cond_real(
            minvalue=1, types=int
        )  # Process delay in unit of ns

        def __post_init__(self) -> None:
            if self.duration is None:
                raise ValueError("Duration must be specified")
            if self.duration < 0:
                raise ValueError("Duration must be positive")
            if self.duration > 100000:
                raise ValueError("Duration must be less than 100000")

    @dataclass
    class DigOut:
        DO_FIFO_UPDATE_PERIOD: int = None
        DO_FIFO_CH_PATTERN: bit_string = bit_string(maxsize=32)
        DO_FIFO_WRITE_COUNT: int = None
        DO_FIFO_WRITE_STARTING_INDEX = None

    @dataclass
    class DigIn:
        DI_UPDATE_PERIOD: int = None

    @dataclass
    class AnaOut:
        AO_UPDATE_PERIOD = None

    @dataclass
    class AnaIn:
        AI_AVG_MODE = None
        AI_UPDATE_PERIOD = None
        AI_BURST_CLOCK_RATE = None
        AI_BURST_CHANNELS = None
        AI_BURST_BUFFER_SIZE = None
        AI_BURST_TRIGGER_MODE = None


class slaveSequenceSetting:
    channel = cond_real(minvalue=0, maxvalue=32, types=int)

    def __init__(self, **kwagrs) -> None:
        _name, _duration = kwagrs.get("name", None), kwagrs.get("duration", None)
        _update_period = kwagrs.get("update_period", None)
        _types = kwagrs.get("types", None)
        _channel = kwagrs.get("channel", None)

        self.__GENERAL = self.general(
            name=_name,
            channel=_channel,
            duration=_duration,
            update_period=_update_period,
            types=_types,
        )

    def todict(self) -> dict:
        return {types: asdict(self.__dict__[types]) for types in self.__dict__.keys()}

    def isnone(self) -> dict:
        none_options = {}
        for types, options in self.todict().items():
            for option, value in options.items():
                if value is None:
                    none_options[types] = option
        return none_options

    def show_options(self) -> dict:
        total_options = {}

        for _, options in self.todict().items():
            for option, value in options.items():
                total_options[option] = value
        return total_options

    @dataclass
    class general:
        types: str = OneOf(None, "DO", "DI", "AO", "AI")
        name: str = None
        channel: int = None
        duration: float = None
        update_period: int = None

        def __post_init__(self) -> None:
            if self.update_period is None:
                raise ValueError("Update period must be specified")
            if self.update_period < 0:
                raise ValueError("Update period must be positive")
            self.length = int(self.duration / self.update_period)

    @property
    def GENERAL(self):
        return self.__GENERAL


class settingValidator:
    # * This class is not used in the current version
    def __init__(
        self, adw: adwinSetting, mas: masterSequenceSetting, slv: slaveSequenceSetting
    ) -> None:
        self.adw = adw
        self.mas = mas
        self.slv = slv

        self.options = self.adw.show_options()
        self.params = self.adw.show_params()

    def validate(self):
        adw_none = self.adw.isnone()
        mas_none = self.mas.isnone()

        if len(adw_none) > 0:
            # for none value options, check if the option is in the master sequence
            for types, options in adw_none:
                # if options in self.mas.show_options(), assign the value to the option
                if options not in self.mas.show_options():
                    raise ValueError(f"Option {options} in {types} is not assigned")

        if len(mas_none) > 0:
            # for none value options, check if the option is in the master sequence
            for types, options in mas_none:
                raise ValueError(f"Option {options} in {types} is not assigned")


class settingAssigner:
    def __init__(self, adw: adwinSetting, mas: masterSequenceSetting) -> None:
        self.adw = adw
        self.mas = mas

    def assign(self):
        # In the ADWIN settings, specify a value that is not the default.
        self._assign_PROCESS_DELAY()
        self._assign_DO_FIFO_CH_PATTERN()
        self._assign_AI_AVG_MODE()  # To Define burst clock rate, AI_AVG_MODE must be assigned first
        # If the value is not specified in the master sequence, assign the default value.
        self._assign_AI_BURST_CLOCK_RATE()
        self._assign_DO_FIFO_WRITE_COUNT()
        self._assign_DO_FIFO_WRITE_STARTING_INDEX()
        self._assign_AO_UPDATE_PERIOD()
        self._assign_AI_BURST_CHANNELS()
        self._assign_AI_BURST_BUFFER_SIZE()
        self._assign_AI_BURST_TRIGGER_MODE()

    def isnone(self) -> dict:
        none_options = {}
        for types, options in self.adw.isnone().items():
            none_options[types] = options
        return none_options

    def show_options(self) -> dict:
        total_options = {}

        for _, options in self.adw.todict().items():
            for option, value in options.items():
                total_options[option] = value
        return total_options

    def _convert_DIO_CH_CONFIG(self):
        """This should be converted to bit"""
        self.adw.GENERAL.DIO_CH_CONFIG = self.adw.GENERAL.DIO_CH_CONFIG

    def _assign_DO_FIFO_CH_PATTERN(self):
        """This should be converted to bit"""
        if self.mas.DO.DO_FIFO_CH_PATTERN is not None:
            self.adw.DO.DO_FIFO_CH_PATTERN = self.mas.DO.DO_FIFO_CH_PATTERN

    def _assign_DO_FIFO_WRITE_COUNT(self):
        if self.mas.DO.DO_FIFO_WRITE_COUNT is not None:
            self.adw.DO.DO_FIFO_WRITE_COUNT = self.mas.DO.DO_FIFO_WRITE_COUNT

    def _assign_DO_FIFO_WRITE_STARTING_INDEX(self):
        if self.mas.DO.DO_FIFO_WRITE_STARTING_INDEX is not None:
            self.adw.DO.DO_FIFO_WRITE_STARTING_INDEX = (
                self.mas.DO.DO_FIFO_WRITE_STARTING_INDEX
            )

    def _assign_AO_UPDATE_PERIOD(self):
        if self.mas.AO.AO_UPDATE_PERIOD is not None:
            self.adw.AO.AO_UPDATE_PERIOD = self.mas.AO.AO_UPDATE_PERIOD

    def _assign_AI_BURST_CLOCK_RATE(self):
        if self.mas.AI.AI_BURST_CLOCK_RATE is not None:
            self.adw.AI.AI_BURST_CLOCK_RATE = self.mas.AI.AI_BURST_CLOCK_RATE
        else:
            min_clock_rates = adwinLimits._MIN_AI_BURST_CLOCK_RATE()
            match self.adw.AI.AI_AVG_MODE:
                case 0:
                    self.adw.AI.AI_BURST_CLOCK_RATE = min_clock_rates.AI_AVG_MODE_0
                case 1:
                    self.adw.AI.AI_BURST_CLOCK_RATE = min_clock_rates.AI_AVG_MODE_1
                case 2:
                    self.adw.AI.AI_BURST_CLOCK_RATE = min_clock_rates.AI_AVG_MODE_2
                case 3:
                    self.adw.AI.AI_BURST_CLOCK_RATE = min_clock_rates.AI_AVG_MODE_3
                case 4:
                    self.adw.AI.AI_BURST_CLOCK_RATE = min_clock_rates.AI_AVG_MODE_4
                case 5:
                    self.adw.AI.AI_BURST_CLOCK_RATE = min_clock_rates.AI_AVG_MODE_5
                case _:
                    raise ValueError("AI_AVG_MODE must be one of 0,1,2,3,4,5")

    def _assign_PROCESS_DELAY(self):
        if self.mas.GENERAL.PROCESS_DELAY is not None:
            self.adw.GENERAL.PROCESS_DELAY = self.mas.GENERAL.PROCESS_DELAY

    def _assign_AI_AVG_MODE(self):
        if self.mas.AI.AI_AVG_MODE is not None:
            self.adw.AI.AI_AVG_MODE = self.mas.AI.AI_AVG_MODE

    def _assign_AI_BURST_CHANNELS(self):
        if self.mas.AI.AI_BURST_CHANNELS is not None:
            self.adw.AI.AI_BURST_CHANNELS = self.mas.AI.AI_BURST_CHANNELS

    def _assign_AI_BURST_BUFFER_SIZE(self):
        if self.mas.AI.AI_BURST_BUFFER_SIZE is not None:
            self.adw.AI.AI_BURST_BUFFER_SIZE = self.mas.AI.AI_BURST_BUFFER_SIZE

    def _assign_AI_BURST_TRIGGER_MODE(self):
        if self.mas.AI.AI_BURST_TRIGGER_MODE is not None:
            self.adw.AI.AI_BURST_TRIGGER_MODE = self.mas.AI.AI_BURST_TRIGGER_MODE
