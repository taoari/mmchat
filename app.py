from utils.gradio import gr_auto_route_pages

demo = gr_auto_route_pages(index_page="index")

if __name__ == "__main__":
    demo.launch()
