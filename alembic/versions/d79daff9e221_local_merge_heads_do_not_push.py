"""LOCAL merge heads (do not push)

Revision ID: d79daff9e221
Revises: 42b4915381e0, e72eff427c95
Create Date: 2025-08-12 17:30:07.565793

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd79daff9e221'
down_revision: Union[str, None] = ('42b4915381e0', 'e72eff427c95')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
