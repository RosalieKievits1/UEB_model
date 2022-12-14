import asyncio
import logging
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
import tifffile as tf
import config

import requests
from requests import Session

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))

def download_dataset_file(
    session: Session,
    base_url: str,
    dataset_name: str,
    dataset_version: str,
    filename: str,
    directory: str,
    overwrite: bool,
) -> Tuple[bool, str]:
    # if a file from this dataset already exists, skip downloading it.
    file_path = Path(directory, filename).resolve()
    if not overwrite and file_path.exists():
        logger.info(f"Dataset file '{filename}' was already downloaded.")
        return True, filename

    endpoint = f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}/files/{filename}/url"
    get_file_response = session.get(endpoint)

    # retrieve download URL for dataset file
    if get_file_response.status_code != 200:
        logger.warning(f"Unable to get file: {filename}")
        logger.warning(get_file_response.content)
        return False, filename

    # use download URL to GET dataset file. We don't need to set the 'Authorization' header,
    # The presigned download URL already has permissions to GET the file contents
    download_url = get_file_response.json().get("temporaryDownloadUrl")
    download_dataset_file_response = requests.get(download_url)

    if download_dataset_file_response.status_code != 200:
        logger.warning(f"Unable to download file: {filename}")
        logger.warning(download_dataset_file_response.content)
        return False, filename

    # write dataset file to disk
    file_path.write_bytes(download_dataset_file_response.content)

    logger.info(f"Downloaded dataset file '{filename}'")
    return True, filename


def list_dataset_files(
    session: Session,
    base_url: str,
    dataset_name: str,
    dataset_version: str,
    params: Dict[str, str],
) -> Tuple[List[str], Dict[str, Any]]:
    logger.info(f"Retrieve dataset files with query params: {params}")

    list_files_endpoint = f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}/files"
    list_files_response = session.get(list_files_endpoint, params=params)

    if list_files_response.status_code != 200:
        raise Exception("Unable to list initial dataset files")


    try:
        list_files_response_json = list_files_response.json()
        dataset_files = list_files_response_json.get("files")
        dataset_filenames = list(map(lambda x: x.get("filename"), dataset_files))
        return dataset_filenames, list_files_response_json
    except Exception as e:
        logger.exception(e)
        raise Exception(e)


async def main():

    # Parameters
    api_key = "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjI4ZWZlOTZkNDk2ZjQ3ZmE5YjMzNWY5NDU3NWQyMzViIiwiaCI6Im11cm11cjEyOCJ9"
    dataset_name = "SVF_NL"
    dataset_version = "3"
    max_keys = "12"

    base_url = "https://api.dataplatform.knmi.nl/open-data/v1"
    "When set to True, if a file with the same name exists the output is written over the file."
    "To prevent unnecessary bandwidth usage, leave it set to False."
    overwrite = False

    download_directory = "/Users/rosaliekievits/Desktop/SVF bestanden MEP"


    "Make sure to send the API key with every HTTP request"
    session = requests.Session()
    session.headers.update({"Authorization": api_key})

    "Verify that the download directory exists"
    if not Path(download_directory).is_dir() or not Path(download_directory).exists():
        raise Exception(f"Invalid or non-existing directory: {download_directory}")

    filenames = []

    start_after_filename = "SVF_r37en2.tif"

    "Use the API to get a list of all dataset filenames"
    #while True:
        # Retrieve dataset files after given filename
    dataset_filenames, response_json = list_dataset_files(
            session,
            base_url,
            dataset_name,
            dataset_version,
            {"maxKeys": f"{max_keys}", "startAfterFilename": start_after_filename},
        )

    "If the result is not truncated, we retrieved all filenames"
    is_truncated = response_json.get("isTruncated")
    if not is_truncated:
        logger.info("Retrieved names of all dataset files")

    # start_after_filename2 = "SVF_r37fz1.tif"
    # #while True:
    #     # Retrieve dataset files after given filename
    # dataset_filenames2, response_json2 = list_dataset_files(
    #         session,
    #         base_url,
    #         dataset_name,
    #         dataset_version,
    #         {"maxKeys": f"{max_keys}", "startAfterFilename": start_after_filename2},
    #     )
    #
    #     # If the result is not truncated, we retrieved all filenames
    # is_truncated = response_json2.get("isTruncated")
    # if not is_truncated:
    #     logger.info("Retrieved names of all dataset files")
    #
    # start_after_filename3 = "SVF_r37fz2.tif"
    # #while True:
    #     # Retrieve dataset files after given filename
    # dataset_filenames3, response_json3 = list_dataset_files(
    #         session,
    #         base_url,
    #         dataset_name,
    #         dataset_version,
    #         {"maxKeys": f"{max_keys}", "startAfterFilename": start_after_filename3},
    #     )
    # # If the result is not truncated, we retrieved all filenames
    # is_truncated = response_json3.get("isTruncated")
    # if not is_truncated:
    #     logger.info("Retrieved names of all dataset files")

    "Store filenames"
    filenames += dataset_filenames
    # filenames += dataset_filenames2
    # filenames += dataset_filenames3

    start_after_filename = dataset_filenames[-1]

    logger.info(f"Number of files to download: {len(filenames)}")
    loop = asyncio.get_event_loop()

    "Allow up to 20 separate threads to download dataset files concurrently"
    executor = ThreadPoolExecutor(max_workers=20)
    futures = []

    "Create tasks that download the dataset files"
    for dataset_filename in filenames:
        "Create future for dataset file"
        future = loop.run_in_executor(
            executor,
            download_dataset_file,
            session,
            base_url,
            dataset_name,
            dataset_version,
            dataset_filename,
            download_directory,
            overwrite,
        )
        futures.append(future)

    "Wait for all tasks to complete and gather the results"
    future_results = await asyncio.gather(*futures)
    print("here")
    logger.info(f"Finished '{dataset_name}' dataset download")

    failed_downloads = list(filter(lambda x: not x[0], future_results))

    if len(failed_downloads) > 0:
        logger.warning("Failed to download the following dataset files:")
        logger.warning(list(map(lambda x: x[1], failed_downloads)))



# if __name__ == "__main__":
#     asyncio.run(main())
""""""

download_directory =  config.input_dir_knmi
SVF_knmi1 = "".join([download_directory, '/SVF_r37hn1.TIF'])
SVF_knmi2 = "".join([download_directory, '/SVF_r37hn2.TIF'])
SVF_knmi3 = "".join([download_directory, '/SVF_r37hz1.TIF'])
SVF_knmi4 = "".join([download_directory, '/SVF_r37hz2.TIF'])

def Verification(SVFs,SVF_knmi, gridboxsize, max_radius, gridboxsize_knmi,matrix):
    """
    :param SVFs: Sky view factors calculated by algorithm
    :param SVF_knmi: sky view factors of knmi
    :param gridboxsize: gridboxsize of calculated sky view factors
    :param max_radius: maximum search radius
    :param gridboxsize_knmi: gridboxsize of knmi sky view factors
    :param matrix: boolean, whether the SVFs are inputted as a matrix
    :return: relative error between the SVF calculated and from KNMI
    """
    "The knmi matrix is based on a different resolution gridboxsize"
    [x_len,y_len] = np.shape(SVF_knmi)
    if gridboxsize==0.5:
        blocklength = int(x_len*y_len)
    elif gridboxsize==5:
        blocklength = int((x_len-2*max_radius/gridboxsize)*(y_len-2*max_radius/gridboxsize))
    dif_array = np.zeros((blocklength))
    rel_dif_array = np.zeros((blocklength))
    idx = 0
    if matrix == True:
        if SVFs.shape != SVF_knmi.shape:
            print("The matrices are not the same shape")
        for i in range(int(x_len)):
            for j in range(int(y_len)):
                """Check what the difference is in SVF"""
                if SVF_knmi[i,j]>=0:
                    dif_array[idx] = SVF_knmi[i,j]-SVFs[i,j]
                    rel_dif_array[idx] = (SVFs[i,j]-SVF_knmi[i,j])/SVFs[i,j]
                    idx += 1
    elif matrix == False:
        """Try 2: we reshape the KNMI one to the list"""
        KNMI_list = []
        for i in range(x_len):
            for j in range(y_len):
                if (gridboxsize==5) and ((max_radius/gridboxsize)<=i and i<(x_len-max_radius/gridboxsize) and (max_radius/gridboxsize)<=j and j<(y_len-max_radius/gridboxsize)):
                    KNMI_list.append(SVF_knmi[i,j])
                if (gridboxsize==0.5) and ((x_len/4)<=i and i<(3*x_len/4) and (y_len/4)<=j and j<(3*y_len/4)):
                    KNMI_list.append(SVF_knmi[i,j])
        if len(KNMI_list) != len(SVFs):
            print("The lists are not the same shape")
        if SVFs is not None:
            KNMI_list = np.array(KNMI_list)
            SVFs = np.array(SVFs)
            dif_array = SVFs[KNMI_list>=0]-KNMI_list[KNMI_list>=0]
            rel_dif_array = (SVFs[KNMI_list>=0]-KNMI_list[KNMI_list>=0])/SVFs[KNMI_list>=0]
            SVF_knmi=KNMI_list

    meanSVFs05 = np.mean(SVFs[SVF_knmi>=0])
    print('The mean of the SVFs computed on 0.5m is ' + str(meanSVFs05))
    print('The variance of the SVFs computed on 0.5 m is ' + str(np.sum(((SVFs[SVF_knmi>=0]-meanSVFs05)**2))/(len(SVFs[SVF_knmi>=0]))))
    meanSVF = np.mean(SVF_knmi[SVF_knmi>=0])
    print('The mean of the SVFs from KNMI is ' + str(meanSVF))
    print('The variance of the SVFs of the KNMI is ' + str(np.sum(((SVF_knmi[SVF_knmi>=0]-meanSVF)**2))/(len(SVF_knmi[SVF_knmi>=0]))))

    """Return the mean of the relative difference"""
    print("The mean relative error is " + str(np.mean(rel_dif_array)*100) + " %")
    print("The maximum absolute error is " + str(np.max(dif_array)))
    print("The maximum relative error is " + str(np.max(rel_dif_array)))
    return

