import asyncio, dataclasses, requests, openai, json, time, argparse, os, httpx
from typing import List, Tuple, Any, Dict, Set, Optional

@dataclasses.dataclass
class Rcpt_Guide_Prompt:
    """A class for generating promp pattern"""

    main_prompt: str
    guest_type: List[str]
    total_words_limit: List[int]
    per_response_limit: List[int]
    specialty: str
    req_options: List[str]
    gt_init: int = 0
    total_init: int = 0
    per_init: int = 0
    opt_init: int = 0

    type_pre: str = "\nGuest type: "
    total_pre: str = "\nTotal words limit: "
    per_pre: str = "\nMinimum words per response: "
    special_pre: str = "\nSpecial requirements: "
    # who_to_ask: str="\nWho to ask: Guest"
    # who_to_answer: str="\nWho to answer: Robo"
    req_pre: str = "\nCurrent request: in the "

    def get_prompts(self):
        strr = self.main_prompt
        i = self.gt_init
        while i < len(self.guest_type):
            j = self.total_init
            while j < len(self.total_words_limit):
                k = self.per_init
                while k < len(self.per_response_limit):
                    m = self.opt_init
                    while m < len(self.req_options):
                        guest = self.type_pre + self.guest_type[i]
                        total = self.total_pre + str(self.total_words_limit[j])
                        per = self.per_pre + str(self.per_response_limit[k])
                        special = self.special_pre + self.specialty
                        req = self.req_pre + self.req_options[m]
                        yield strr + guest + total + per + special + req, f"gt-{i}-total-{j}-per-{k}-opt-{m}"
                        m += 1
                    k += 1
                j += 1
            i += 1

