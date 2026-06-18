from fastmcp import FastMCP
from server.tools.kb_tools import mcp as kb_mcp
from server.tools.crm_tools import mcp as crm_mcp
from server.tools.ticket_tools import mcp as ticket_mcp

mcp = FastMCP("UDA-Hub Tool Server")

mcp.mount("kb",kb_mcp)
mcp.mount("crm",crm_mcp)
mcp.mount("tickets",ticket_mcp)

if __name__ == "__main__":
    mcp.run()
