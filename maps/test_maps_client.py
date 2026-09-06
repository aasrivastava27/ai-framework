from maps_client import MapsClient

def main():
    maps = MapsClient()

    # ---------------------------------------------------------
    # Geocode
    # ---------------------------------------------------------

    print("\n--- GEOCODE ---")

    result = maps.geocode_location(
        "Gateway of India, Mumbai"
    )

    print(result)

    # ---------------------------------------------------------
    # Route
    # ---------------------------------------------------------

    print("\n--- ROUTE ---")

    result = maps.get_route(
        "Gateway of India, Mumbai",
        "Bandra Fort, Mumbai",
    )

    print(result)

    # ---------------------------------------------------------
    # Distance
    # ---------------------------------------------------------

    print("\n--- DISTANCE ---")

    result = maps.get_distance(
        "Gateway of India, Mumbai",
        "Bandra Fort, Mumbai",
    )

    print(result)

    # ---------------------------------------------------------
    # Nearby places
    # ---------------------------------------------------------

    print("\n--- NEARBY PLACES ---")

    result = maps.find_nearby_places(
        "Gateway of India, Mumbai",
        "cafe",
        radius_m=1000,
        limit=5,
    )

    print(result)


if __name__ == "__main__":
    main()
