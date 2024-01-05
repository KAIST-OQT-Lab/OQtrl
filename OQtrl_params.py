from dataclasses import dataclass, asdict, field
from OQtrl_descriptors import bit_string
from ctypes import c_uint32

UNIT_TIME = 1e-9  # 1ns
DO_UNIT_TIME = 1e-9 * 10  # 10ns


@dataclass(frozen=True)
class adwinLimits:
    AO_MIN_base10: int = -10
    AO_MAX_base10: int = 10
    AI_AVG_MODE_0: int = 25
    AI_AVG_MODE_1: int = 67
    AI_AVG_MODE_2: int = 154
    AI_AVG_MODE_3: int = 313
    AI_AVG_MODE_4: int = 645
    AI_AVG_MODE_5: int = 1333


@dataclass(init=True)
class plotParams:
    FIG_SIZE: tuple = (5, 3)
    DPI: int = 600
    LINEWIDTH: int = 2
    RECT: list = field(default_factory=list)  # left, bottom, width, height
    FONT_SIZE: int = 10  # font size for axis label

    def __post_init__(self):
        self.RECT = [0, 0, 1.2, 0.4]

    def as_dict(self):
        return asdict(self)


@dataclass(init=True)
class generalParams:
    """General settings class

    Attributes:
        DIO_CH_CONFIG: bit_string = bit_string(maxsize=32)
        DURATION: int = None
        EXPERIMENT_MODE: int = None
    """

    DIO_CH_CONFIG: bit_string = bit_string("0001", maxsize=4)
    DURATION: int = None
    EXPERIMENT_MODE: int = None

    def as_dict(self):
        return asdict(self)


@dataclass(init=True)
class digOutParams:
    """Digital output settings class

    Attributes:
        DO_FIFO_CH_PATTERN: bit_string = bit_string(maxsize=32)
        DO_FIFO_WRITE_COUNT: int = None
        DO_FIFO_WRITE_STARTING_INDEX = None
    """

    DO_FIFO_CH_PATTERN: bit_string = bit_string(maxsize=32)
    DO_FIFO_WRITE_COUNT: int = None
    DO_FIFO_WRITE_STARTING_INDEX: int = 1

    def as_dict(self):
        return asdict(self)


@dataclass(init=True)
class digOutDatas:
    """Digital output FIFO data

    Attributes:
        DO_FIFO_PATTERN: c_uint32 = None
    """

    DO_FIFO_PATTERN: c_uint32 = None

    def as_dict(self):
        return asdict(self)


@dataclass(init=True)
class digInParams:
    """Digital input settings class

    Attributes:
        DI_UPDATE_PERIOD: int = None
    """

    DI_UPDATE_PERIOD: int = None

    def as_dict(self):
        return asdict(self)


@dataclass(init=True)
class anaOutParams:
    """Analog output settings class

    Attributes:
        AO_UPDATE_PERIOD = defaults to 3000
    """

    AO_UPDATE_PERIOD: int = 3000

    def as_dict(self):
        return asdict(self)


@dataclass(init=True)
class anaOutDatas:
    """Analog output data

    Attributes:
        AO_PATTERN: c_uint32 = None
    """
    #TODO:
    #AO_PATTERN: c_uint32 = None
     
    def as_dict(self):
        return asdict(self)


@dataclass(init=True)
class anaInParams:
    """Analog input settings class

    Attributes:
        AI_AVG_MODE = None
        AI_UPDATE_PERIOD = None
        AI_BURST_CLOCK_RATE = None
        AI_BURST_CHANNELS = None
        AI_BURST_BUFFER_SIZE = None
        AI_BURST_TRIGGER_MODE = None
    """

    AI_AVG_MODE: int = 0
    AI_UPDATE_PERIOD: int = 1
    AI_BURST_CLOCK_RATE: int = 20000
    AI_BURST_CHANNELS: int = None
    AI_BURST_BUFFER_SIZE: int = None
    AI_BURST_TRIGGER_MODE: int = None

    def as_dict(self):
        return asdict(self)


@dataclass(init=True, frozen=True)
class paramReferNum:
    """Assigned parameters class

    Attributes:
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
    """

    DIO_CH_CONFIG: int = 1
    DURATION: int = 2
    EXPERIMENT_MODE: int = 11
    DO_FIFO_CH_PATTERN: int = 31
    DO_FIFO_WRITE_COUNT: int = 32
    DO_FIFO_WRITE_STARTING_INDEX: int = 33
    AI_AVG_MODE: int = 40
    AI_BURST_CHANNELS: int = 41
    AI_BURST_BUFFER_SIZE: int = 42
    AI_BURST_CLOCK_RATE: int = 43
    AI_BURST_TRIGGER_MODE: int = 44
    AO_UPDATE_PERIOD: int = 50

    def as_dict(self):
        return asdict(self)


@dataclass(init=True, frozen=True)
class dataReferNum:
    """Assigned datas class

    Attributes:
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
    """

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

    def as_dict(self):
        return asdict(self)


class adwinParams:
    def __init__(self):
        self.generalParams = generalParams()
        self.dig_out_params = digOutParams()
        self.dig_out_datas = digOutDatas()

        self.dig_in_params = digInParams()

        self.ana_out_params = anaOutParams()
        self.ana_out_datas = anaOutDatas()

        self.ana_in_params = anaInParams()

    def as_dict(self):
        # As_dict each attribute and then combine them into a single dictionary.
        return {
            **self.generalParams.as_dict(),
            **self.dig_out_params.as_dict(),
            **self.dig_out_datas.as_dict(),
            **self.dig_in_params.as_dict(),
            **self.ana_out_params.as_dict(),
            **self.ana_out_datas.as_dict(),
            **self.ana_in_params.as_dict(),
        }
    