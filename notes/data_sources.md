# where the numbers come from

Property records and closed sales come from the City of Philadelphia's Office of Property Assessment (OPA), published as open data through a public SQL endpoint. Each row is a parcel with its attributes (beds, baths, livable area, lot, year built, condition, quality grade) and its most recent recorded sale price and date, plus the city's own assessed market value.

Recorded sales lag the closing by four to eight weeks because deeds are recorded after settlement. The last two months in any pull are therefore thin and should not be read as a slowdown.

Sales under fifty thousand dollars are excluded. Most of them are transfers between family members, sheriff sales or partial-interest deeds, not arm's-length transactions, and they would poison a median.

County market indicators come from Redfin's Data Center market tracker, which aggregates MLS data monthly by county and property type. For Philadelphia the "All Residential" series should be used: Redfin files rowhomes under "Townhouse", so its "Single Family" series for the city is a small detached-only slice.

None of this is MLS data. An MLS listing carries a list price, days on market, status history and agent remarks; public records carry none of those. Where the agent needs a list price it uses the last recorded sale or a price supplied by the user, and where it needs DOM it uses the county median unless the user supplies the listing's own.
