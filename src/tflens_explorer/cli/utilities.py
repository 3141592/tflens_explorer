

def parse_kv_args(args: list[str]) -> dict:
    result = {}

    for arg in args:
        if "=" not in arg:
            continue

        key, value = arg.split("=", 1)

        # basic type coercion
        if value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass  # leave as string

        result[key] = value

    return result

def get_shape(shape_str):
    index1 = shape_str.find('[')
    index2 = shape_str.find(']')

    if index1 >= 0 and index2 >= 0:
        shape = shape_str[index1:index2 + 1]
    else:
        shape = shape_str
    return shape