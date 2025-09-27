from pathlib import Path

def load_namespace():
    here = Path(__file__).parent
    parts = []
    parts.append((here / 'imagine_part1.py').read_text(encoding='utf-8'))
    parts.append((here / 'imagine_part2.py').read_text(encoding='utf-8'))
    parts.append((here / 'imagine_part3.py').read_text(encoding='utf-8'))
    parts.append((here / 'imagine_part4.py').read_text(encoding='utf-8'))
    parts.append((here / 'imagine_part5.py').read_text(encoding='utf-8'))
    parts.append((here / 'imagine_part6.py').read_text(encoding='utf-8'))
    parts.append((here / 'imagine_part7.py').read_text(encoding='utf-8'))
    full_src = ''.join(parts)
    ns = {}
    ns['__file__'] = str(here / "imagine_combined.py")
    code = compile(full_src, str(here / "imagine_combined.py"), 'exec')
    exec(code, ns, ns)
    return ns

def run_gui():
    ns = load_namespace()
    for entry in ('main', 'run', 'start'):
        fn = ns.get(entry)
        if callable(fn):
            return fn()
    App = ns.get('ImagineApp')
    if App is None:
        for k, v in ns.items():
            if isinstance(v, type) and k.lower().endswith('app'):
                App = v
                break
    if App is None:
        raise RuntimeError('Could not find ImagineApp class')

    # Prefer the new PySide6 experience when available
    try:
        from PySide6 import QtWidgets  # type: ignore
    except Exception:
        QtWidgets = None  # type: ignore

    if QtWidgets is not None and isinstance(App, type) and issubclass(App, QtWidgets.QWidget):
        qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = App()
        window.show()
        qt_app.exec()
        return

    import tkinter as tk
    root = tk.Tk()
    app = App(root)
    root.mainloop()
