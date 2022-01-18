def post_processor_factory(post_processor_input_settings):
    if post_processor_input_settings['type'].lower() == 'exodus':
        from .exodus_post_processor import ExodusPostProcessor
        return ExodusPostProcessor(post_processor_input_settings)
    else:
        assert False, 'Unsupported post processor type!'
