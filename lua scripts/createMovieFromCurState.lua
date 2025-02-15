-- Creates state loaded m64 from current state using rest of currently playing TAS

function replaceBytes(inp, offset, val, num_bytes)
	local val_str = ""
	for i=1,num_bytes do
		val_str = val_str .. string.char((val >> (8 * (i-1))) & 0xFF)
	end
	return inp:sub(1, offset) .. val_str .. inp:sub(offset + num_bytes + 1)
end

function createStateLoadedMovie()
	-- Get base m64 data
	local src_m64_path = movie.get_filename()
	local cur_frame = emu.samplecount()

	local src_m64_file = io.open(src_m64_path, "rb")
	if not src_m64_file then
		print("Currently playing m64 no longer exists.")
		return
	end
	local m64_header = src_m64_file:read(0x400)
	src_m64_file:seek("cur", 4*cur_frame)
	local input_data = src_m64_file:read("*all")
	src_m64_file:close()

	-- Update header
	local length = #input_data >> 2
	local cur_time = os.time(os.date("!*t")) -- should prevent uid overlap desyncing frame count
	m64_header = replaceBytes(m64_header, 0x08, cur_time, 4) -- uid
	m64_header = replaceBytes(m64_header, 0x0C, -1, 4) -- vi count
	m64_header = replaceBytes(m64_header, 0x18, length, 4) -- frame count
	m64_header = replaceBytes(m64_header, 0x1C, 1, 2) -- state loaded flag

	-- Get new paths
	local new_m64_path = iohelper.filediag("*.m64", 1)
	if (new_m64_path:sub(-4) ~= ".m64") then
		print("Invalid path (should end with .m64).")
		return
	end

	-- Savestate filename
	local first_period_index = new_m64_path:find("%.")
	local new_st_path = new_m64_path:sub(1, first_period_index) .. "st"
	
	-- Save files
	local new_m64_file = io.open(new_m64_path, "wb")
	new_m64_file:write(m64_header)
	new_m64_file:write(input_data)
	new_m64_file:close()

	savestate.savefile(new_st_path)
end

createStateLoadedMovie()