from argparse import ArgumentParser
from src.data.process.process_satellite import process_satellite
from src.data.process.build_dataframe_from_sat import main as build_dataframe_from_sat
from src.data.process.save_intensities_and_motion_hdf import main as save_intensities_and_motion_hdf
from src.data.process.calc_rain_events_datetimes import main as calc_rain_events_datetimes

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("rain_events_file", type=str)
    parser.add_argument("--location", type=str, default="rio_de_janeiro")
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()
    # process raw satellite data
    process_satellite(args.rain_events_file,
                      location=args.location, num_workers=args.num_workers)
    # build dataframe from processed satellite data
    build_dataframe_from_sat(
        args.rain_events_file, location=args.location, num_workers=args.num_workers)

    # calculate datetimes for intensities and motion fields
    rain_datetimes = calc_rain_events_datetimes(
        args.rain_events_file, exclude_first=10, exclude_last=0)
    save_intensities_and_motion_hdf(
        datetimes=rain_datetimes, location=args.location)
