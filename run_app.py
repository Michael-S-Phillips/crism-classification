"""Entry point: python run_app.py --img <mrral.img> --gpkg <mineral_map.gpkg>"""
import argparse, uvicorn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',  required=True, help='Path to mrral .img file')
    parser.add_argument('--gpkg', required=True, help='Path to mineral_map .gpkg')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8765, type=int)
    args = parser.parse_args()

    import os
    os.environ['CRISM_IMG']  = args.img
    os.environ['CRISM_GPKG'] = args.gpkg

    uvicorn.run('app.main:app', host=args.host, port=args.port, reload=False)

if __name__ == '__main__':
    main()
