



class PseudoVarFloat:
    """
    Variable class for float values.
    Inputs:
        val (float): The input float value.
    Outputs:
        value (float): The float value.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "val": ("FLOAT", {
                    "default": 0.5
                })
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("flt",)
    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Vars"

    def func(self, val):
        flt = float(val)
        print(f"[pseudocomfy] PseudoVarFloat: {flt}")
        return (flt,)


class PseudoVarInt:
    """
    Variable class for integer values.
    Inputs:
        val (int): The input integer value.
    Outputs:
        value (int): The integer value.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "val": ("INT", {
                    "default": 0
                })
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Vars"

    def func(self, val):
        i = int(val)
        print(f"[pseudocomfy] PseudoVarInt: {i}")
        return (i,)


class PseudoVarString:
    """
    Variable class for string values.
    Inputs:
        val (str): The input string value.
    Outputs:
        value (str): The string value.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "val": ("STRING", {
                    "default": "",
                    "multiline": True
                })
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("str",)
    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Vars"

    def func(self, val):
        s = str(val)
        print(f"[pseudocomfy] PseudoVarString: {s}")
        return (s,)


class PseudoSeed:
    """
    Seed node. Produces an integer seed value to feed noise/sampling.

    Unlike the PseudoVar* nodes this is not a variable: it is meant to be
    dropped onto the canvas wherever a seed is needed and driven externally
    (e.g. by the Pseudorandom Rhino plugin) rather than hand edited.

    The widget is named "seed" so ComfyUI automatically attaches its
    standard control_after_generate (fixed / increment / decrement /
    randomize) behaviour, and so external tooling can set the value via the
    node's widgets_values in the submitted workflow.

    Inputs:
        seed (int): The seed value (0 .. 2**64 - 1).
    Outputs:
        seed (int): The seed value, passed through.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                })
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "func"
    CATEGORY = "Pseudocomfy/Vars"

    def func(self, seed):
        seed = int(seed)
        print(f"[pseudocomfy] PseudoSeed: {seed}")
        return (seed,)

