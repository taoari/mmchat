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


def gr_auto_route(pages, index_page="index"):
    """Dynamically generates routes for Gradio UI components from the pages module."""
    with gr.Blocks() as demo:
        # Load the index page first (default entry page)
        if hasattr(pages, index_page):
            page_module = getattr(pages, index_page)
            if hasattr(page_module, "demo"):
                page_module.demo.render()

    # Iterate through available page modules, excluding special attributes and the index page
    for module_name in dir(pages):
        if module_name.startswith("__") or module_name == index_page:
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
