import spatialdata
from spatialdata import SpatialData
from tqdm import tqdm
import re
import logging
import pathlib
import pandas
import numpy

log = logging.getLogger(__name__)

def spatialdata_concat(
    sdata_path_list: list[str],
    save_path: str,
    table_only: bool = False,
    drop_cols: list = None,
    output_obs_csv: bool = False,
    ) -> SpatialData:

    concatenated_sdata = {}

    for sd_path in tqdm(sdata_path_list, desc="Loading SpatialData"):
        sd_path = pathlib.Path(sd_path)

        log.info(f"Loading {sd_path.name}.")

        try:
            if table_only:
                loop_sdata = spatialdata.read_zarr(sd_path, selection=["tables", "table"])
            else:
                loop_sdata = spatialdata.read_zarr(sd_path)

            batch_id = loop_sdata["table"].obs["batch"][0]

            # We need to add this for concatenate to work. Here, we are using
            # concatenation as a method of storing data together for processing,
            # rather than relying on functionality with plotting etc. so conventions
            # are skipped.
            # We therefore do not need accurate region/instance keys, so we set
            # them to the default Xenium uses.
            # I'm unsure why our zarr conversion has failed to include the spatialdata_attrs
            loop_sdata.tables["table"].obs["region"] = "cell_circles"
            loop_sdata.tables["table"].uns["spatialdata_attrs"] = {} 
            loop_sdata.tables["table"].uns["spatialdata_attrs"]["region"] = "cell_circles"
            loop_sdata.tables["table"].uns["spatialdata_attrs"]["region_key"] = "region"
            loop_sdata.tables["table"].uns["spatialdata_attrs"]["instance_key"] = "cell_id"

            concatenated_sdata[batch_id] = loop_sdata
        except Exception as e:
            log.error(f"Unable to read: {sd_path}. Error: {e}. SpatialData will be skipped.")

    print(f"Concatenating spatial data...")
    concatenated_sdata = spatialdata.concatenate(
        concatenated_sdata, 
        region_key="region",
        instance_key="region",
        concatenate_tables=True, 
        modify_tables_inplace=True, # For speed
        join="outer",
    )

    if drop_cols is not None:
        if isinstance(drop_cols, str):
            drop_cols = [drop_cols]
        
        concatenated_sdata["table"].obs = concatenated_sdata["table"].obs.drop(columns=drop_cols)

    # For writing to zarr we need to fill string object columns nan. However, for object float
    # columns we do not need to fill na. 
    for col in concatenated_sdata["table"].obs.select_dtypes(include="object").columns:
        if concatenated_sdata["table"].obs[col].dropna().map(type).eq(str).all():
            # Only fill NaNs if the non-null values are all strings
            concatenated_sdata["table"].obs[col] = concatenated_sdata["table"].obs[col].fillna("NaN")

    print(f"Writing spatial data...")
    concatenated_sdata.write(save_path, overwrite = True)

    if output_obs_csv:
        csv_filename = str(pathlib.Path(save_path).parent / pathlib.Path(save_path).stem) + ".csv"

        concatenated_sdata["table"].obs.to_csv(csv_filename)


