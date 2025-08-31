import py_compile
try:
    py_compile.compile(r"D:/PulsaTalker/app.py", doraise=True)
    print("OK")
except Exception as e:
    print("CompileError:", e)
