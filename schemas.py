"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List

# Example schemas (kept for reference):

class User(BaseModel):
    """
    Users collection schema
    Collection name: "user" (lowercase of class name)
    """
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    """
    Products collection schema
    Collection name: "product" (lowercase of class name)
    """
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# SaaS PDF chat app schemas

class Document(BaseModel):
    """Stores a user's uploaded document metadata"""
    user_id: Optional[str] = Field(None, description="Owner of the document")
    filename: str = Field(..., description="Original filename")
    pages: int = Field(..., ge=0, description="Number of pages")
    chunk_count: int = Field(0, ge=0, description="Number of text chunks")
    status: str = Field("ready", description="processing status: ready|processing|error")

class Chunk(BaseModel):
    """Chunk of a document"""
    doc_id: str = Field(..., description="Reference to the document _id as string")
    text: str = Field(..., description="Chunk text")
    index: int = Field(..., ge=0, description="Chunk index order")
    # Embedding optional; we compute on the fly for lightweight runtime
    embedding: Optional[List[float]] = Field(None, description="Optional precomputed embedding")

class ChatMessage(BaseModel):
    """Stores chat messages for analytics/audit (optional)"""
    doc_id: str
    role: str = Field(..., description="user|assistant|system")
    content: str

# Note: The Flames database viewer will automatically:
# 1. Read these schemas from GET /schema endpoint
# 2. Use them for document validation when creating/editing
# 3. Handle all database operations (CRUD) directly
# 4. You don't need to create any database endpoints!
