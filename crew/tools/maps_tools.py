from crewai.tools import BaseTool
from maps.maps_client import MapsClient
from pydantic import BaseModel, Field
from typing import Type


maps_client  =  MapsClient()

# =========================================================

# Geocode Location

# =========================================================

class GeocodeInput(BaseModel):


    location: str  =  Field(
        description = "The location or address to find."
    )


class GeocodeLocationTool(BaseTool):


    name: str  =  "Geocode Location"

    description: str  =  (
        "Find the geographic coordinates and OpenStreetMap "
        "information for a location or address."
    )

    args_schema: Type[BaseModel]  =  GeocodeInput

    def _run(self, location: str):

        return maps_client.geocode_location(
            location
        )


# =========================================================

# Get Route

# =========================================================

class RouteInput(BaseModel):


    origin: str  =  Field(
        description = "Starting location."
    )

    destination: str  =  Field(
        description = "Destination location."
    )

    profile: str  =  Field(
        default = "car",
        description = "Travel mode: car, foot, or bike."
    )


class GetRouteTool(BaseTool):


    name: str  =  "Get Route"

    description: str  =  (
        "Calculate a route between two locations. "
        "Supports car, walking, and bicycle routes."
    )

    args_schema: Type[BaseModel]  =  RouteInput

    def _run(
        self,
        origin: str,
        destination: str,
        profile: str  =  "car",
    ):

        return maps_client.get_route(
            origin = origin,
            destination = destination,
            profile = profile,
        )


# =========================================================

# Get Distance

# =========================================================

class DistanceInput(BaseModel):


    origin: str  =  Field(
        description = "Starting location."
    )

    destination: str  =  Field(
        description = "Destination location."
    )


class GetDistanceTool(BaseTool):


    name: str  =  "Get Distance"

    description: str  =  (
        "Calculate the driving distance and estimated "
        "travel duration between two locations."
    )

    args_schema: Type[BaseModel]  =  DistanceInput

    def _run(
        self,
        origin: str,
        destination: str,
    ):

        return maps_client.get_distance(
            origin = origin,
            destination = destination,
        )


# =========================================================

# Find Nearby Places

# =========================================================

class NearbyPlacesInput(BaseModel):


    location: str  =  Field(
        description = "Location around which to search."
    )

    category: str  =  Field(
        description = (
            "Type of place to search for, such as "
            "cafe, restaurant, hospital, pharmacy, hotel."
        )
    )

    radius_m: int  =  Field(
        default = 1000,
        description = "Search radius in meters."
    )

    limit: int  =  Field(
        default = 10,
        description = "Maximum number of places to return."
    )


class FindNearbyPlacesTool(BaseTool):


    name: str  =  "Find Nearby Places"

    description: str  =  (
        "Find nearby places of a specified category "
        "using OpenStreetMap."
    )

    args_schema: Type[BaseModel]  =  NearbyPlacesInput

    def _run(
        self,
        location: str,
        category: str,
        radius_m: int  =  1000,
        limit: int  =  10,
    ):

        return maps_client.find_nearby_places(
            location = location,
            category = category,
            radius_m = radius_m,
            limit = limit,
        )


# =========================================================

# Get Place Details

# =========================================================

class PlaceDetailsInput(BaseModel):
    osm_id: str  =  Field(
        description = (
            "OpenStreetMap ID such as "
            "node / 123456789."
        )
    )


class GetPlaceDetailsTool(BaseTool):

    name: str  =  "Get Place Details"

    description: str  =  (
        "Get detailed information about an OpenStreetMap "
        "place using its OSM ID."
    )

    args_schema: Type[BaseModel]  =  PlaceDetailsInput

    def _run(self, osm_id: str):

        return maps_client.get_place_details(
            osm_id
        )
