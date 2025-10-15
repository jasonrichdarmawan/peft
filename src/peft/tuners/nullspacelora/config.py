from peft.utils import PeftType
from peft.tuners.lora import LoraConfig

class NullSpaceLoraConfig(LoraConfig):
    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.NULLSPACELORA