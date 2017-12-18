def _start_shell(local_ns=None, global_ns=None):
    # An interactive shell is useful for debugging/development.
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
        user_ns.update(global_ns)
        IPython.start_ipython(argv=[], user_ns=user_ns)

