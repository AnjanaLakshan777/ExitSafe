"""Keywords used to recognise each event type.

``EventType`` itself is part of the stored event schema
(app.data.schemas.event_schema).
"""

from app.data.schemas.event_schema import EventType


# Checked in this order; the first type with a matching keyword wins, so more
# specific types come before broader ones (a data breach is also a
# cybersecurity incident, but DATA_BREACH is the more useful label).
EVENT_KEYWORDS = {
    EventType.FRAUD_OR_GOVERNANCE: ["fraud", "embezzlement", "misappropriation",
                                    "accounting irregularit", "forensic audit", "whistleblower"],
    EventType.DATA_BREACH: ["data breach", "data leak", "data exposure", "customer data",
                            "records exposed", "personal data"],
    EventType.CYBERSECURITY_INCIDENT: ["cyber attack", "cyberattack", "ransomware", "hack",
                                       "malware", "cybersecurity incident", "system outage"],
    EventType.CREDIT_EVENT: ["default", "downgrade", "credit rating", "restructuring of debt",
                             "missed payment", "insolvency"],
    EventType.REGULATORY_ACTION: ["regulator", "penalty", "fined", "sanction", "suspension",
                                  "central bank directive", "license revoked"],
    EventType.LEGAL_EVENT: ["lawsuit", "court", "litigation", "legal action", "arbitration"],
    EventType.LIQUIDITY_EVENT: ["trading halt", "trading suspended", "delisting",
                                "bank run", "withdrawal restrictions"],
    EventType.MANAGEMENT_CHANGE: ["resigns", "resignation", "appointed", "appointment",
                                  "chief executive", "ceo", "chairman", "board of directors"],
    EventType.FINANCIAL_RESULT: ["quarterly results", "annual results", "interim financial",
                                 "profit", "loss", "earnings", "revenue", "dividend"],
    EventType.MACROECONOMIC_EVENT: ["inflation", "interest rate", "policy rate", "exchange rate",
                                    "gdp", "sovereign", "imf"],
    EventType.SECTOR_EVENT: ["sector", "industry-wide", "tariff", "import ban"],
}
