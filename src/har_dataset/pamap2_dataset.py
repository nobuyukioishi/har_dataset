from typing import List, Tuple, Dict, Union, Literal

from pathlib import Path, PosixPath
import os
import errno
import requests
from tqdm import tqdm
from .utils import get_file_list_in_zip, linear_interpolation
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import warnings


class Pamap2Dataset:
    # Consts
    IMU_SENSOR_SAMPLE_RATE = 100  # 100 Hz
    HR_SENSOR_SAMPLE_RATE = 9  # ~9 Hz
    TIMESTAMP_COLUMN_NAME = "timestamp"
    LABEL_COLUMN_NAME = "activityID"
    SENSOR_TYPES: Tuple[str] = ("acc8g", "acc16g", "gyro", "mag")
    SUBJECTS: Tuple[str] = tuple(f"subject{i}" for i in range(101, 110))
    RUNS: Tuple[str] = ("Protocol", "Optional")
    IMU_SENSORS: Tuple[str] = ("IMU_hand", "IMU_chest", "IMU_ankle")

    ACTIVITY_ID_DICT: Dict[int, str] = {
        1: "lying",
        2: "sitting",
        3: "standing",
        4: "walking",
        5: "running",
        6: "cycling",
        7: "Nordic walking",
        9: "watching TV",
        10: "computer work",
        11: "car driving",
        12: "ascending stairs",
        13: "descending stairs",
        16: "vacuum cleaning",
        17: "ironing",
        18: "folding laundry",
        19: "house cleaning",
        20: "playing soccer",
        24: "rope jumping",
        0: "other",
    }

    ZIP_FILE_NAME: str = "PAMAP2_Dataset.zip"
    DEFAULT_DATASET_DIR: str = "./data/"
    ARCHIVE_URL: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"

    dataset_path = None

    @staticmethod
    def download_dataset(
        to: Union[Path, str] = "./data/", archive_url: str = ARCHIVE_URL
    ):
        """
        Download the PAMAP2 dataset to the given path
        :param to: directory the dataset zip file is to be saved
        :param archive_url:
        :return:
        """
        if type(to) != PosixPath:
            to = Path(to)

        if not to.is_dir():
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                f"{to} not found. Please specify a directory that already exists.",
            )

        res = requests.get(archive_url, stream=True)
        chunk_size = 1024
        with open(to / Pamap2Dataset.ZIP_FILE_NAME, "wb") as f:
            total_length = int(res.headers.get("content-length"))
            for chunk in tqdm(
                res.iter_content(chunk_size=chunk_size),
                total=int(total_length / chunk_size) + 1,
                desc=f"Downloading {Pamap2Dataset.ZIP_FILE_NAME}...",
            ):
                if chunk:
                    f.write(chunk)
                    f.flush()

            print(f"{Pamap2Dataset.ZIP_FILE_NAME} has been saved in {to}")

        return res

    @staticmethod
    def get_imu_data_columns(
        imu_sensors: List[Literal[IMU_SENSORS]] = list(IMU_SENSORS),
        sensor_types: List[str] = ["acc16g", "gyro", "mag"],
    ):
        """Return data columns of the specified IMU sensor(s). In PAMAP2 dataset, accelerometer data has been recorded
         in two different scales, ±16g and ±6g. Since some data exceeds the range of ±6g, this library uses the data
         recorded in ±16g by default
        :param imu_sensors: List of IMU sensors
        :param sensor_types: subset of Pamap2Dataset.SENSOR_TYPES
        :return:
        """
        return [
            f"{imu_sensor}_{col}"
            for imu_sensor in imu_sensors
            for col in [
                f"{sensor}{axis}"
                for sensor in sensor_types
                for axis in ["X", "Y", "Z"]
            ]
        ]

    @staticmethod
    def get_col_list():
        """
        Return the column names of the PAMAP2 dataset
        ----------
        column_no. description
        ----------
        1. temperature (Â°C)
        2-4. 3D-acceleration data (ms-2), scale: Â±16g, resolution: 13-bit
        5-7. 3D-acceleration data (ms-2), scale: Â±6g, resolution: 13-bit # not enough for some activities
        8-10. 3D-gyroscope data (rad/s)
        11-13. 3D-magnetometer data (Î¼T)
        14-17. orientation (invalid in this data collection)
        ----------

        Parameters
        ----------
        Returns list of columns of the PAMAP2 dataset
        -------
        """

        imu_cols = (
            ["temp"]
            + [
                f"{sensor}{axis}"
                for sensor in ["acc16", "acc6", "gyro", "mag"]
                for axis in ["X", "Y", "Z"]
            ]
            + [f"_quat{axis}" for axis in ["W", "X", "Y", "Z"]]  # not used
        )
        return ["timestamp", "activityID", "heart rate"] + [
            f"{imu_sensor}_{imu_col}"
            for imu_sensor in Pamap2Dataset.IMU_SENSORS
            for imu_col in imu_cols
        ]

    def __init__(
        self,
        dataset_dir: Union[Path, str] = DEFAULT_DATASET_DIR,
        download: bool = False,
    ):
        """
        :param dataset_dir:
        """

        if type(dataset_dir) != PosixPath:
            dataset_dir = Path(dataset_dir)

        # Check if `dataset_dir` directory exists.
        if not dataset_dir.is_dir():
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                f"{dataset_dir} {'(Default directory)' if dataset_dir == self.DEFAULT_DATASET_DIR else ''} not found. "
                f"Please specify a directory that already exists.",
            )

        dataset_path = dataset_dir / self.ZIP_FILE_NAME

        # Check if the dataset file exists.
        if not dataset_path.is_file():
            # Download the dataset when download is True
            if download:
                # Check if dataset_dir exists
                self.download_dataset(to=dataset_dir)
            else:
                raise FileNotFoundError(
                    errno.ENOENT,
                    os.strerror(errno.ENOENT),
                    f"{dataset_path} (default path) not found. Specify `dataset_dir` in the constructor that contains "
                    f"{self.ZIP_FILE_NAME} or download the dataset file in `dataset_dir` by passing `download=True` "
                    f"to the constructor. The default `dataset_dir` is '{self.DEFAULT_DATASET_DIR}'",
                )
        self.dataset_path = dataset_path

    def get_filepath_of_sbj_run_in_zip(
        self,
        subject: Literal[SUBJECTS],
        run: Literal[RUNS] = "Protocol",
    ) -> str:
        """
        Return the filepath of the specified run of the specified subject.
        :param run: "Protocol" or "Optional" (of Pamap2Dataset.RUNS)
        :param subject: one of Pamap2Dataset.SUBJECTS
        :return:
        """

        file_path = f"PAMAP2_Dataset/{run}/{subject}.dat"

        # Check if file exists in zip.
        assert file_path in get_file_list_in_zip(self.dataset_path)

        return file_path

    def load_file(
        self,
        subject: Literal[SUBJECTS],
        run: Literal[RUNS] = "Protocol",
        fillna: bool = False,
        set_timestamp_as_index: bool = False,
    ) -> pd.DataFrame:
        """
        Load a sensor data file of the specified run of the specified subject and return the data in the DataFrame
        format.
        :param subject:
        :param run:
        :param fillna:
        :param set_timestamp_as_index:
        :return: data
        """

        # prepare file_path and opp_columns
        file_path = self.get_filepath_of_sbj_run_in_zip(subject, run)
        opp_columns = self.get_col_list()

        # load .dat file and covert it to dataframe
        with ZipFile(self.dataset_path, "r") as opp_zip:
            b_data = opp_zip.read(file_path)
            df_data = pd.read_csv(
                BytesIO(b_data), sep=" ", header=None, names=opp_columns
            )

        if fillna:
            df_data = linear_interpolation(df_data)
            if df_data.isnull().values.any():
                warnings.warn("nan values still remain in the data")

        if set_timestamp_as_index:
            df_data = df_data.set_index(Pamap2Dataset.TIMESTAMP_COLUMN_NAME)

        return df_data
