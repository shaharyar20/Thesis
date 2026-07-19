from torchlbm.state import TorchlbmState
from pathlib import Path
import torch
from typing import List
import numpy as np
import pandas as pd
import os
import cv2

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import torchlbm.standalone_operations.file_operations as file_o
from torchlbm.logger import Logger
from torchlbm.io_tools.vtk_output_writer import get_single_node_output_data
from torchlbm.io_tools.pdf_output_writer import get_single_node_pyplot_data
from torchlbm.io_tools.pytorch_output_writer import get_single_node_pytorch_data
from vtk import vtkXMLImageDataWriter, vtkXMLPolyDataWriter


class OutputWriter:
    """Writes output for a simulation."""

    def __init__(
        self,
        result_folder: Path,
        logger: Logger,
        state: TorchlbmState,
    ) -> None:
        """Initializer for the ouptout writer.

        Args:
            result_folder (Path): The result folder, where the results are written to.
            logger (Logger): The logger to write information to the terminal and the log file
            state (TorchlbmState): The state that contains all information about the simulation.
        """
        self._result_folder = result_folder
        self._vtk_folder = self._result_folder.joinpath("output")
        file_o.create_folder(self._vtk_folder)
        self._visualization_folder = self._result_folder.joinpath("visualization")
        file_o.create_folder(self._visualization_folder)
        self._pytorch_output_folder = self._result_folder.joinpath("pytorch_output")
        file_o.create_folder(self._pytorch_output_folder)
        self.__logger = logger
        self.__dimension = state.torchlbm_setup["Domain"]["Dimension"].value

        self._image_lists = {
            "velocity": [],
            "density": [],
        }
        # (time, filename) of every .vti written, for the ParaView collection file.
        self._pvd_entries = []

    def _write_pvd(self) -> None:
        """Write/refresh ``output/output.pvd`` -- a ParaView collection indexing every .vti.

        ParaView only auto-groups a file series when the names look like ``name_<digits>.ext``.
        Our .vti names embed the float simulation time (``output_352.25121656.vti``), so the
        digits are not a clean sequence and ParaView loads each file as a SEPARATE dataset.
        A .pvd sidesteps the naming rule entirely and, unlike renaming to a zero-padded index,
        it also carries the REAL timestep -- so ParaView's time slider shows physical time
        instead of a file counter.

        Open ``output.pvd`` in ParaView (not the individual .vti files) to get one dataset
        with a working time slider. Rewritten after every output, so it stays valid even if
        the run is interrupted.
        """
        pvd_path = self._vtk_folder.joinpath("output.pvd")
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            "  <Collection>",
        ]
        for time_value, file_name in self._pvd_entries:
            lines.append(
                f'    <DataSet timestep="{time_value:.8f}" group="" part="0" file="{file_name}"/>'
            )
        lines += ["  </Collection>", "</VTKFile>"]
        pvd_path.write_text("\n".join(lines) + "\n")

    def write_output(self, state: TorchlbmState, timestamp: int) -> None:
        """Writes output to files.

        Args:
            state (TorchlbmState): The state of the simulation that contains all relevant information, including
                               the fields.
            timestamp (int): The currect timestep.
        """
        if torch.cuda.is_available():
            state.cpu()
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            state.cpu()

        internal_cells_list = state.torchlbm_setup["Domain"]["InternalCells"].value
        num_halos = state.torchlbm_setup["Domain"]["NumHaloCells"].value
        dimension = state.torchlbm_setup["Domain"]["DimensionInteger"].value
        delta_x = state.lattice_distance

        time_string = f"{timestamp:.8f}"
        vtk_file = f"output_{time_string}.vti"
        vtk_filename = self._vtk_folder.joinpath(vtk_file)
        output_data = get_single_node_output_data(state.node_data, state.unit_converter, internal_cells_list, num_halos, dimension)
        output_data.SetSpacing(delta_x, delta_x, delta_x)
        writer = vtkXMLImageDataWriter()
        writer.SetDataModeToBinary()
        self.__logger.write(f"Write: {str(vtk_filename)}")
        self.__logger.star_line_flush()
        writer.SetFileName(str(vtk_filename))
        writer.SetInputData(output_data)
        writer.Write()
        # Index this .vti in the ParaView collection so the series groups (see _write_pvd).
        self._pvd_entries.append((float(timestamp), vtk_file))
        self._write_pvd()

        pyplot_figures = get_single_node_pyplot_data(state=state)
        for key, value in pyplot_figures.items():
            pdf_filename = self._visualization_folder.joinpath(f"{key}_{time_string}.pdf")
            value.savefig(pdf_filename)
            png_filename = self._visualization_folder.joinpath(f"{key}_{time_string}.png")
            value.savefig(png_filename, dpi=200)
            plt.close(value)
            if key in self._image_lists.keys():
                self._image_lists[key].append(str(png_filename.absolute()))
        plt.close("all")

        pytorch_data = get_single_node_pytorch_data(state=state)
        for key, value in pytorch_data.items():
            pytorch_filename = self._pytorch_output_folder.joinpath(f"{key}_{time_string}.pt")
            torch.save(value, pytorch_filename)

        if torch.cuda.is_available():
            state.cuda()
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            state.mps()

    def get_artifacts(self, state: TorchlbmState, timestamp: int):
        if torch.cuda.is_available():
            state.cpu()
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            state.cpu()

        time_string = f"{timestamp:.8f}"
        artifacts = {}

        pyplot_figures = get_single_node_pyplot_data(state=state)
        for key, value in pyplot_figures.items():
            artifacts[f"{key}_{time_string}.png"] = value

        if torch.cuda.is_available():
            state.cuda()
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            state.mps()

        return artifacts

    def generate_videos(self, state: TorchlbmState):
        if state.torchlbm_setup["Output"]["Velocity"]["Active"].value:
            for key, value in self._image_lists.items():
                frame = cv2.imread(str(self._image_lists[key][0]))
                height, width, layers = frame.shape

                video = cv2.VideoWriter(str(self._visualization_folder.joinpath(f"{key}.mov")), cv2.VideoWriter_fourcc("m", "p", "4", "v"), 15, (width, height))

                for image in self._image_lists[key]:
                    video.write(cv2.imread(str(image)))
                cv2.destroyAllWindows()
                video.release()
                # import moviepy.video.io.ImageSequenceClip
                # movie_clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(value, 15)
                # movie_clip.write_videofile(str(self._visualization_folder.joinpath(f"{key}.mov")))
