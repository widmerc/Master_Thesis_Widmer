# =============================================================================
# QGIS Processing Script: Compute School-Route Alternatives (Safety-Aware Routing)
# =============================================================================
# Author: Claude Widmer
# Master Thesis, University of Zurich (GIUZ), 2025
# Supervisors: Prof. Ross Purves, Dr. Tumasch Reichen
#
# Description:
# ------------
# This script integrates QGIS with the School Route Safety Routing API developed
# in the master's thesis "School Route Safety: A Computer Vision Framework for
# Automated Evaluation in Zurich" (Claude Widmer, University of Zurich, 2025).
#
# It enables interactive calculation of safety-aware pedestrian routes that
# combine network length and automated safety scores derived from computer-
# vision analysis. The tool supports two routing modes:
#
#   - Default mode: Computes four predefined routes (Fastest, Balanced, Safe, Safest)
#   - Custom mode:  Allows manual tuning of α (distance) and β (safety) parameters
#
# The algorithm handles coordinate transformation (LV95 / EPSG:2056), API
# communication, and automatic styling of results. Output routes are stored as
# GeoJSON, GPKG, or SHP layers.
# =============================================================================

import json, os, tempfile, urllib.request, urllib.error, shutil
from qgis.core import (
    QgsProcessingAlgorithm, QgsProcessingParameterPoint, QgsProcessingParameterString,
    QgsProcessingParameterEnum, QgsProcessingParameterVectorDestination,
    QgsProcessingException, QgsVectorLayer, QgsProject,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform,
    QgsVectorFileWriter, QgsLineSymbol, QgsUnitTypes
)

class LoadRouteAlgorithm(QgsProcessingAlgorithm):
    START = "START_POINT"                          
    END = "END_POINT"                              
    URL = "API_URL"                                
    MODEL = "MODEL"                                
    ALPHA = "ALPHA"                                
    BETA = "BETA"                                  
    OUTPUT = "OUTPUT"                              

    # -------------------------------------------------------------------------
    def name(self): 
        return "compute_school_route_alternatives"

    def displayName(self): 
        return "Compute School-Route Alternatives (Safety-Aware Routing)"

    def group(self): 
        return "Routing"

    def groupId(self): 
        return "routing"

    # -------------------------------------------------------------------------
    def shortHelpString(self):
        """Shown in the QGIS Processing Toolbox (right panel)."""
        return (
            "<b>Compute School-Route Alternatives (Safety-Aware Routing)</b><br><br>"
            "Developed as part of the master’s thesis <i>“School Route Safety: A Computer Vision Framework "
            "for Automated Evaluation in Zurich”</i> (Claude Widmer, University of Zurich, 2025).<br><br>"

            "This tool connects QGIS with the custom <b>School Route Safety Routing API</b>, "
            "enabling the calculation of <b>safety-aware pedestrian routes</b>.<br><br>"

            "<b>Modes:</b><br>"
            "<ul>"
            "<li><b>Default mode:</b> Computes four predefined route types (Fastest, Balanced, Safe, Safest).</li>"
            "<li><b>Custom mode:</b> Allows manual definition of α (distance) and β (safety) weights.</li>"
            "</ul><br>"

            "Coordinates are automatically transformed to <b>LV95 (EPSG:2056)</b>.<br>"
            "Output routes are styled (red line, 0.35 mm) and saved as GeoJSON, GPKG, or SHP.<br><br>"

            "<b>Purpose:</b><br>"
            "Explore how safety-based routing changes optimal school routes and visualize "
            "safer path alternatives directly in QGIS."
        )

    # -------------------------------------------------------------------------
    def initAlgorithm(self, config=None):
        """Define input parameters for the tool."""
        self.addParameter(QgsProcessingParameterPoint(self.START, "Start point"))
        self.addParameter(QgsProcessingParameterPoint(self.END, "End point"))

        self.addParameter(QgsProcessingParameterString(
            self.URL, "Routing API URL",
            defaultValue="http://localhost:8000/route",
        ))

        self.addParameter(QgsProcessingParameterEnum(
            self.MODEL, "Routing model", options=["ML", "Rule-based"], defaultValue=0
        ))

        self.addParameter(QgsProcessingParameterString(
            self.ALPHA, "Alpha (optional)", defaultValue="", optional=True
        ))
        self.addParameter(QgsProcessingParameterString(
            self.BETA, "Beta (optional)", defaultValue="", optional=True
        ))

        self.addParameter(QgsProcessingParameterVectorDestination(
            self.OUTPUT, "Output (LV95 / EPSG:2056)"
        ))

    # -------------------------------------------------------------------------
    def processAlgorithm(self, parameters, context, feedback):
        """Main execution logic of the routing request."""
        start_pt = self.parameterAsPoint(parameters, self.START, context)
        end_pt   = self.parameterAsPoint(parameters, self.END, context)
        url      = self.parameterAsString(parameters, self.URL, context)
        model_idx = self.parameterAsEnum(parameters, self.MODEL, context)
        model = "ml" if model_idx == 0 else "rule-based"

        alpha_str = self.parameterAsString(parameters, self.ALPHA, context) or ""
        beta_str  = self.parameterAsString(parameters, self.BETA, context) or ""
        dest_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        alpha, beta = None, None
        if alpha_str.strip():
            try:
                alpha = float(alpha_str)
            except ValueError:
                raise QgsProcessingException("Alpha must be numeric.")
        if beta_str.strip():
            try:
                beta = float(beta_str)
            except ValueError:
                raise QgsProcessingException("Beta must be numeric.")
        if (alpha is None) != (beta is None):
            raise QgsProcessingException("Provide both α and β or leave both empty.")

        # ---------------------------------------------------------------------
        lv95 = QgsCoordinateReferenceSystem("EPSG:2056")
        project_crs = QgsProject.instance().crs() or lv95
        if project_crs != lv95:
            ct = QgsCoordinateTransform(project_crs, lv95, QgsProject.instance().transformContext())
            start_lv95, end_lv95 = ct.transform(start_pt), ct.transform(end_pt)
        else:
            start_lv95, end_lv95 = start_pt, end_pt

        payload = {"start": [start_lv95.x(), start_lv95.y()],
                   "end": [end_lv95.x(), end_lv95.y()],
                   "model": model}
        if alpha is not None and beta is not None:
            payload.update({"alpha": alpha, "beta": beta})

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        mode = "Custom α/β" if alpha else "Default 4 cases"
        feedback.pushInfo(f"➡️ Requesting {url} ({mode}, model={model})")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                geojson_str = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise QgsProcessingException(f"HTTP error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise QgsProcessingException(f"URL error: {e.reason}")
        except Exception as e:
            raise QgsProcessingException(f"Unexpected API error: {e}")

        fd, tmp_path = tempfile.mkstemp(suffix=".geojson"); os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(geojson_str)

        tmp_copy = tmp_path.replace(".geojson", "_copy.geojson")
        shutil.copy(tmp_path, tmp_copy)

        src_layer = QgsVectorLayer(tmp_copy, "routes_lv95", "ogr")
        if not src_layer.isValid():
            os.unlink(tmp_path); os.unlink(tmp_copy)
            raise QgsProcessingException("Invalid GeoJSON from API.")
        if src_layer.crs() != lv95:
            src_layer.setCrs(lv95)

        _, ext = os.path.splitext(dest_path or ""); ext = ext.lower()
        if ext in (".geojson", ".json"): driver = "GeoJSON"
        elif ext == ".gpkg": driver = "GPKG"
        elif ext == ".shp": driver = "ESRI Shapefile"
        else:
            driver, dest_path = "GPKG", dest_path or tempfile.mktemp(suffix=".gpkg")

        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName, opts.fileEncoding = driver, "UTF-8"

        result = QgsVectorFileWriter.writeAsVectorFormatV2(
            src_layer, dest_path, QgsProject.instance().transformContext(), opts
        )
        os.unlink(tmp_path)
        if result[0] != QgsVectorFileWriter.NoError:
            raise QgsProcessingException(f"Write failed: {result[1]}")

        out_layer = QgsVectorLayer(dest_path, "school_routes_lv95", "ogr")
        symbol = QgsLineSymbol.createSimple({
            "color": "255,0,0", "width": "0.35",
            "width_unit": str(QgsUnitTypes.RenderMillimeters)
        })
        out_layer.renderer().setSymbol(symbol)
        out_layer.triggerRepaint()
        out_layer.saveDefaultStyle()

        feedback.pushInfo(f"✅ Output saved to: {dest_path}")
        return {self.OUTPUT: dest_path}

    # -------------------------------------------------------------------------
    def createInstance(self): 
        return LoadRouteAlgorithm()

# Factory method for QGIS Processing
def classFactory(iface): 
    return LoadRouteAlgorithm()
