

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

