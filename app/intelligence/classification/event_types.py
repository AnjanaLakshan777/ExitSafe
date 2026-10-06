"""Keywords used to recognise each event type."""

from app.data.schemas.event_schema import EventType


# Checked in order and the first match wins, so specific types (a data breach)
# come before broader ones (a cybersecurity incident).
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
