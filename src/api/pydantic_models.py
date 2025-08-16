
from pydantic import BaseModel

class CreditRiskInput(BaseModel):
    """Input features for credit risk prediction."""
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    PricingStrategy: int
    FraudResult: int
    Amount_sum: float
    Amount_mean: float
    Amount_std: float
    Amount_count: int
    TransactionId_count: int
    hour: int
    day: int
    month: int
    year: int
    day_of_week: int

class CreditRiskOutput(BaseModel):
    """Output for credit risk prediction."""
    risk_probability: float
