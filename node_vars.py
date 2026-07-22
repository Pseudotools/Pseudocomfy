



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

