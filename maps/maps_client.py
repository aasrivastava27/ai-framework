import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MapsClient:


    def __init__(self):

        self.server_params = StdioServerParameters(
            command="node",
            args=[
                "/Users/aashutosh/Documents/ai_framework/maps/osm-mcp-main/dist/index.js"
            ],
        )

    # ---------------------------------------------------------
    # Call an MCP tool
    # ---------------------------------------------------------

    async def _call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ):

        async with stdio_client(
            self.server_params
        ) as (read, write):

            async with ClientSession(
                read,
                write
            ) as session:

                await session.initialize()

                result = await session.call_tool(
                    tool_name,
                    arguments=arguments,
                )

                if result.isError:
                    raise RuntimeError(
                        f"MCP tool '{tool_name}' failed."
                    )

                return result.structuredContent

    # ---------------------------------------------------------
    # Geocode a location
    # ---------------------------------------------------------

    def geocode_location(
        self,
        location: str,
    ):

        result = asyncio.run(
            self._call_tool(
                "geocode",
                {
                    "query": location,
                    "countrycodes": "in",
                    "limit": 1,
                },
            )
        )

        results = result.get(
            "results",
            []
        )

        if not results:
            raise ValueError(
                f"Could not find location: {location}"
            )

        return results[0]

    # ---------------------------------------------------------
    # Calculate route
    # ---------------------------------------------------------

    def get_route(
        self,
        origin: str,
        destination: str,
        profile: str = "car",
    ):

        result = asyncio.run(
            self._call_tool(
                "route",
                {
                    "waypoints": [
                        origin,
                        destination,
                    ],
                    "profile": profile,
                },
            )
        )

        return result

    # ---------------------------------------------------------
    # Calculate distance
    # ---------------------------------------------------------

    def get_distance(
        self,
        origin: str,
        destination: str,
    ):

        route = self.get_route(
            origin,
            destination,
        )

        distance_meters = route.get(
            "distance_m",
            0,
        )

        duration_seconds = route.get(
            "duration_s",
            0,
        )

        return {
            "origin": origin,
            "destination": destination,
            "distance_km": round(
                distance_meters / 1000,
                2,
            ),
            "duration_minutes": round(
                duration_seconds / 60,
                1,
            ),
        }

    # ---------------------------------------------------------
    # Find nearby places
    # ---------------------------------------------------------

    def find_nearby_places(
        self,
        location: str,
        category: str,
        radius_m: int = 1000,
        limit: int = 10,
    ):

        result = asyncio.run(
            self._call_tool(
                "find_nearby_pois",
                {
                    "near": location,
                    "category": category,
                    "radius_m": radius_m,
                    "limit": limit,
                },
            )
        )

        return result

    # ---------------------------------------------------------
    # Get place details
    # ---------------------------------------------------------

    def get_place_details(
        self,
        osm_id: str,
    ):

        result = asyncio.run(
            self._call_tool(
                "poi_details",
                {
                    "osm_id": osm_id,
                },
            )
        )

        return result

