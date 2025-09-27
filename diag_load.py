from load_parts import load_namespace

ns = load_namespace()
print('Loaded namespace keys:', sorted(k for k in ns.keys() if not k.startswith('__')))
if '_fallback_tracebacks' in ns:
    print('\nFallback tracebacks present:')
    for i, tb in enumerate(ns['_fallback_tracebacks']):
        print(f'--- traceback {i+1} ---')
        print(tb)
else:
    print('\nNo fallback tracebacks; primary compile succeeded.')
    # Check presence of a few expected functions/classes
    for name in ['ImagineApp', 'run', 'apply_duotone', 'liquify_deform', 'swirl_image']:
        print(name, '->', 'OK' if name in ns else 'MISSING')
