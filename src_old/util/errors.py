
def general_tardigrade_error(error_message):
    try:
        assert False
    except AssertionError:
        raise Exception(error_message)