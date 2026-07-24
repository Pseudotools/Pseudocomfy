



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

