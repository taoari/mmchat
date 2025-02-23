import gradio as gr
import pathlib


# Function to dynamically load all Python modules from a given directory
def load_modules(path: str):
    """Loads all .py files from the specified directory as Python modules."""
    pages_dir = pathlib.Path(path)
    loaded_modules = []

    if pages_dir.exists():
        for py_file in pages_dir.glob("*.py"):
            module_name = f"{path}.{py_file.stem}"  # Use fully qualified module name
            module = __import__(module_name, fromlist=[None])
            loaded_modules.append(module)

    return loaded_modules


def gr_auto_route(pages, index_page="index", includes=[], excludes=[]):
    """Dynamically generates routes for Gradio UI components from the pages module."""

    # Default to list all modules in pages if includes is empty
    if not includes:
        includes = dir(pages)

    with gr.Blocks() as demo:
        # Load the index page first (default entry page)
        if hasattr(pages, index_page):
            page_module = getattr(pages, index_page)
            if hasattr(page_module, "demo"):
                page_module.demo.render()

    # Iterate through the provided includes, excluding specified modules and the index page
    for module_name in includes:
        if (
            module_name.startswith("__")
            or module_name == index_page
            or module_name in excludes
        ):
            continue

        page_module = getattr(pages, module_name)
        if hasattr(page_module, "demo"):
            with demo.route(module_name.replace("_", " ").title()):
                page_module.demo.render()

    return demo


def gr_auto_route_pages(index_page="index"):
    "Gradio auto route apps under 'pages/' directory"
    import pages

    load_modules("pages")
    demo = gr_auto_route(pages, index_page=index_page)
    return demo
