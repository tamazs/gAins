from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.post("/", response_model=OutputModel)
# async def converter(input: InputModel) -> OutputModel:
#     prompt: str = input.request
#
#     agent = ConversionAgent()
#
#     response = agent.run(prompt)
#
#     output_model = OutputModel(response=response)
#     return output_model


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)