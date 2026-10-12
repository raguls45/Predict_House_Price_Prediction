from __future__ import annotations

import json
import math
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "web"
MODELS_DIR = PROJECT_ROOT / "models"
FEATURE_COLUMNS = [
    "X1 transaction date",
    "X2 house age",
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude",
]

model = joblib.load(MODELS_DIR / "house_price_model.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
selector = joblib.load(MODELS_DIR / "selector.pkl")


class PredictionHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 16_384:
                raise ValueError("Please submit all property details.")

            payload: Any = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("Property details must be sent as an object.")

            values: dict[str, float] = {}
            for field in FEATURE_COLUMNS:
                if field not in payload:
                    raise ValueError(f"Missing property detail: {field}.")
                try:
                    value = float(payload[field])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Enter a valid number for {field}.") from error
                if not math.isfinite(value):
                    raise ValueError(f"Enter a finite number for {field}.")
                values[field] = value

            if values["X2 house age"] < 0:
                raise ValueError("House age cannot be negative.")
            if values["X3 distance to the nearest MRT station"] < 0:
                raise ValueError("MRT distance cannot be negative.")
            stores = values["X4 number of convenience stores"]
            if stores < 0 or not stores.is_integer():
                raise ValueError("Convenience stores must be a non-negative whole number.")
            if not -90 <= values["X5 latitude"] <= 90:
                raise ValueError("Latitude must be between -90 and 90.")
            if not -180 <= values["X6 longitude"] <= 180:
                raise ValueError("Longitude must be between -180 and 180.")

            property_data = pd.DataFrame([values], columns=FEATURE_COLUMNS)
            scaled = scaler.transform(property_data)
            selected = selector.transform(scaled)
            predicted_price = float(model.predict(selected)[0])

            self._send_json(
                HTTPStatus.OK,
                {
                    "predicted_price": round(predicted_price, 2),
                    "unit": "dataset price units per unit area",
                },
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Request body must be valid JSON."})
        except ValueError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Prediction failed. Check that model files are available."})

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


class HousePriceServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main() -> None:
    import os
    from functools import partial
    from http.server import SimpleHTTPRequestHandler

    handler = partial(SimpleHTTPRequestHandler, directory=str(WEB_DIR))
    handler.__name__ = "FrontendHandler"

    class CombinedHandler(PredictionHandler, handler):  # type: ignore[misc, valid-type]
        """Serve the frontend files and prediction endpoint from one origin."""

    server = HousePriceServer(("127.0.0.1", int(os.environ.get("PORT", "8000"))), CombinedHandler)
    print("House Price Studio running at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down House Price Studio.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
