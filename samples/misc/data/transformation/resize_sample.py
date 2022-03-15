from pathlib import Path

from data.synthesis.resize.resize_children import resize_children

if __name__ == "__main__":

    def aush():
        src_path = (
            Path.home()
            / "ProjectsWin"
            / "Github"
            / "Aivclab"
            / "eyetest"
            / "images2"
            / "faces"
            / "raw"
        )
        resize_children(src_path, 512)

    aush()
